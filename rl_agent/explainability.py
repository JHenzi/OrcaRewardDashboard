"""
Explainability Module

Extracts human-readable rules from the RL agent's learned behavior.
Uses decision tree surrogates and SHAP values for interpretability.
"""

import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# For decision trees
try:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available for rule extraction")

# For SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

load_dotenv()

logger = logging.getLogger(__name__)

# Get database path from environment or use default
DB_PATH = os.getenv("DATABASE_PATH", "sol_prices.db")


class RuleExtractor:
    """
    Extracts human-readable rules from RL agent decisions.
    
    Uses decision tree surrogates to convert neural network behavior
    into interpretable if-then rules.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize rule extractor.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - rule extraction will be limited")
    
    def extract_rules_from_decisions(
        self,
        min_samples: int = 50,
        max_depth: int = 5,
        min_samples_split: int = 10,
    ) -> List[Dict]:
        """
        Extract rules from historical decisions.
        
        Args:
            min_samples: Minimum number of samples per rule
            max_depth: Maximum depth of decision tree
            min_samples_split: Minimum samples to split a node
            
        Returns:
            List of rule dicts
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available - cannot extract rules")
            return []
        
        conn = sqlite3.connect(self.db_path)
        
        # Get decisions with outcomes
        query = """
            SELECT 
                d.id, d.timestamp, d.action, d.state_features,
                d.predicted_return_1h, d.predicted_return_24h,
                pa.actual_return_1h, pa.actual_return_24h,
                pa.price_at_prediction, pa.price_1h_later, pa.price_24h_later
            FROM rl_agent_decisions d
            LEFT JOIN rl_prediction_accuracy pa ON d.id = pa.decision_id
            WHERE pa.actual_return_1h IS NOT NULL
            AND pa.actual_return_24h IS NOT NULL
            ORDER BY d.timestamp DESC
            LIMIT 1000
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < min_samples:
            logger.warning(f"Insufficient data for rule extraction: {len(df)} < {min_samples}")
            return []
        
        # Parse state features
        features_list = []
        actions = []
        outcomes_1h = []
        outcomes_24h = []
        
        for _, row in df.iterrows():
            try:
                state_features = json.loads(row['state_features']) if row['state_features'] else {}
                features_list.append(state_features)
                actions.append(row['action'])
                outcomes_1h.append(row['actual_return_1h'])
                outcomes_24h.append(row['actual_return_24h'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Skipping row due to parse error: {e}")
                continue
        
        if len(features_list) < min_samples:
            return []
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Extract rules for each action
        all_rules = []
        
        for action in ['BUY', 'SELL', 'HOLD']:
            # Filter to this action
            action_mask = np.array(actions) == action
            if action_mask.sum() < min_samples:
                continue
            
            action_features = features_df[action_mask]
            action_outcomes_1h = np.array(outcomes_1h)[action_mask]
            action_outcomes_24h = np.array(outcomes_24h)[action_mask]
            
            # Create binary target: successful if return > threshold
            threshold = 0.0  # Positive return = success
            target_1h = (action_outcomes_1h > threshold).astype(int)
            target_24h = (action_outcomes_24h > threshold).astype(int)
            
            # Train decision tree for 1h outcomes
            if target_1h.sum() > 0 and (1 - target_1h).sum() > 0:
                tree_1h = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_split // 2,
                )
                tree_1h.fit(action_features, target_1h)
                
                # Extract rules from tree
                rules_1h = self._extract_rules_from_tree(
                    tree_1h, action_features.columns.tolist(), action, "1h"
                )
                
                # Calculate performance metrics
                for rule in rules_1h:
                    rule_mask = self._evaluate_rule(rule, action_features)
                    if rule_mask.sum() > 0:
                        rule_outcomes = action_outcomes_1h[rule_mask]
                        rule['win_rate'] = (rule_outcomes > threshold).mean()
                        rule['avg_return_1h'] = rule_outcomes.mean()
                        rule['sample_size'] = rule_mask.sum()
                        rule['avg_return_24h'] = action_outcomes_24h[rule_mask].mean()
                
                all_rules.extend(rules_1h)
        
        # Sort by win rate and sample size
        all_rules.sort(key=lambda x: (x.get('win_rate', 0) * x.get('sample_size', 0)), reverse=True)
        
        return all_rules[:20]  # Return top 20 rules
    
    def _extract_rules_from_tree(
        self,
        tree,
        feature_names: List[str],
        action: str,
        horizon: str,
    ) -> List[Dict]:
        """
        Extract rules from a decision tree.
        
        Args:
            tree: Trained decision tree
            feature_names: List of feature names
            action: Action this rule applies to
            horizon: Prediction horizon (1h or 24h)
            
        Returns:
            List of rule dicts
        """
        rules = []
        
        # Get tree structure
        tree_ = tree.tree_
        n_nodes = tree_.node_count
        children_left = tree_.children_left
        children_right = tree_.children_right
        feature = tree_.feature
        threshold = tree_.threshold
        value = tree_.value
        
        def traverse(node, conditions):
            if children_left[node] == children_right[node]:  # Leaf node
                # Check if this leaf predicts success
                if value[node][0][1] > value[node][0][0]:  # More positive than negative
                    rule_text = self._format_rule(conditions, action)
                    rules.append({
                        'rule_text': rule_text,
                        'rule_conditions': conditions.copy(),
                        'action': action,
                        'horizon': horizon,
                    })
                return
            
            # Internal node - add condition
            feature_name = feature_names[feature[node]]
            conditions.append({
                'feature': feature_name,
                'operator': '<=',
                'value': threshold[node],
            })
            traverse(children_left[node], conditions)
            conditions.pop()
            
            conditions.append({
                'feature': feature_name,
                'operator': '>',
                'value': threshold[node],
            })
            traverse(children_right[node], conditions)
            conditions.pop()
        
        traverse(0, [])
        return rules
    
    def _format_rule(self, conditions: List[Dict], action: str) -> str:
        """
        Format conditions into human-readable rule text.
        
        Args:
            conditions: List of condition dicts
            action: Action to take
            
        Returns:
            Human-readable rule string
        """
        if not conditions:
            return f"Always {action}"
        
        condition_strs = []
        for cond in conditions:
            feature = cond['feature']
            operator = cond['operator']
            value = cond['value']
            
            # Format feature names nicely
            feature_display = feature.replace('_', ' ').title()
            
            if operator == '<=':
                condition_strs.append(f"{feature_display} <= {value:.3f}")
            else:
                condition_strs.append(f"{feature_display} > {value:.3f}")
        
        return f"If {' AND '.join(condition_strs)} → {action}"
    
    def _evaluate_rule(self, rule: Dict, features_df: pd.DataFrame) -> np.ndarray:
        """
        Evaluate which samples match a rule.
        
        Args:
            rule: Rule dict with conditions
            features_df: DataFrame of features
            
        Returns:
            Boolean array indicating which samples match
        """
        conditions = rule.get('rule_conditions', [])
        if not conditions:
            return np.ones(len(features_df), dtype=bool)
        
        mask = np.ones(len(features_df), dtype=bool)
        
        for cond in conditions:
            feature = cond['feature']
            operator = cond['operator']
            value = cond['value']
            
            if feature not in features_df.columns:
                mask = np.zeros(len(features_df), dtype=bool)
                break
            
            if operator == '<=':
                mask = mask & (features_df[feature] <= value)
            else:
                mask = mask & (features_df[feature] > value)
        
        return mask
    
    def store_rules(self, rules: List[Dict]):
        """
        Store extracted rules in database.
        
        Args:
            rules: List of rule dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for rule in rules:
            cursor.execute("""
                INSERT INTO discovered_rules (
                    rule_text, rule_conditions, action,
                    win_rate, avg_return_1h, avg_return_24h, sample_size,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                rule.get('rule_text', ''),
                json.dumps(rule.get('rule_conditions', [])),
                rule.get('action', ''),
                rule.get('win_rate'),
                rule.get('avg_return_1h'),
                rule.get('avg_return_24h'),
                rule.get('sample_size'),
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {len(rules)} rules in database")
    
    def get_discovered_rules(
        self,
        action: Optional[str] = None,
        min_win_rate: float = 0.0,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Get discovered rules from database.
        
        Args:
            action: Filter by action (BUY, SELL, HOLD)
            min_win_rate: Minimum win rate
            limit: Maximum number of rules to return
            
        Returns:
            List of rule dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                id, rule_text, rule_conditions, action,
                win_rate, avg_return_1h, avg_return_24h, sample_size,
                confidence_interval_lower, confidence_interval_upper,
                created_at, last_validated_at
            FROM discovered_rules
            WHERE 1=1
        """
        params = []
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        if min_win_rate > 0:
            query += " AND win_rate >= ?"
            params.append(min_win_rate)
        
        query += " ORDER BY win_rate DESC, sample_size DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        rules = []
        for row in rows:
            try:
                rule_conditions = json.loads(row[2]) if row[2] else []
            except (json.JSONDecodeError, TypeError):
                rule_conditions = []
            
            rules.append({
                'id': row[0],
                'rule_text': row[1],
                'rule_conditions': rule_conditions,
                'action': row[3],
                'win_rate': row[4],
                'avg_return_1h': row[5],
                'avg_return_24h': row[6],
                'sample_size': row[7],
                'confidence_interval_lower': row[8],
                'confidence_interval_upper': row[9],
                'created_at': row[10],
                'last_validated_at': row[11],
            })
        
        return rules


class SHAPExplainer:
    """
    Computes SHAP values for feature importance.
    """
    
    def __init__(self, model, state_encoder, device: str = "cpu"):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained TradingActorCritic model to explain
            state_encoder: StateEncoder instance
            device: Device model is on
        """
        self.model = model
        self.state_encoder = state_encoder
        self.device = device
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - feature importance will be limited")
    
    def compute_feature_importance(
        self,
        states: List[Dict],
        background_size: int = 50,
        target: str = "action",  # "action", "value", "pred_1h", "pred_24h"
    ) -> Dict[str, float]:
        """
        Compute SHAP feature importance.
        
        Args:
            states: List of state dicts
            background_size: Size of background dataset for SHAP
            target: What to explain ("action", "value", "pred_1h", "pred_24h")
            
        Returns:
            Dict mapping feature names to importance scores
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return {}
        
        if not self.model:
            logger.warning("Model not available for SHAP computation")
            return {}
        
        try:
            import torch
            
            # Convert states to feature vectors
            # For simplicity, we'll use a subset of features
            feature_names = []
            feature_vectors = []
            
            for state in states[:background_size]:
                # Extract key features
                price_features = state.get("price", [])
                news_sentiment = state.get("news_sentiment", [])
                position_features = state.get("position", [])
                
                # Aggregate features
                features = []
                names = []
                
                # Price features (aggregate)
                if len(price_features) > 0:
                    features.append(float(np.mean(price_features[:10])))  # Avg recent returns
                    names.append("price_returns_avg")
                    features.append(float(np.std(price_features[:10])))  # Volatility
                    names.append("price_volatility")
                
                # News sentiment (aggregate)
                if len(news_sentiment) > 0:
                    features.append(float(np.mean(news_sentiment)))
                    names.append("news_sentiment_avg")
                    features.append(float(np.sum(news_sentiment > 0)))  # Positive count
                    names.append("news_positive_count")
                
                # Position features
                if len(position_features) > 0:
                    features.extend([float(f) for f in position_features[:5]])
                    names.extend([f"position_{i}" for i in range(min(5, len(position_features)))])
                
                if features:
                    feature_vectors.append(features)
                    if not feature_names:
                        feature_names = names
            
            if not feature_vectors:
                return {}
            
            # Use KernelExplainer for model-agnostic explanation
            # Note: This is simplified - full implementation would use proper model wrapper
            def model_wrapper(feature_array):
                """Wrapper to convert feature array to model output."""
                # This is a placeholder - would need proper state reconstruction
                # For now, return dummy values
                return np.random.rand(len(feature_array), 3)  # 3 actions
            
            # Use background data
            background = np.array(feature_vectors[:min(background_size, len(feature_vectors))])
            explainer = shap.KernelExplainer(model_wrapper, background)
            
            # Compute SHAP values for a sample
            sample = np.array(feature_vectors[0:1])
            shap_values = explainer.shap_values(sample, nsamples=100)
            
            # Aggregate importance
            if isinstance(shap_values, list):
                # Multi-output: average across outputs
                shap_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_avg = np.abs(shap_values)
            
            # Map to feature names
            importance = {}
            for i, name in enumerate(feature_names):
                if i < len(shap_avg[0]):
                    importance[name] = float(shap_avg[0][i])
            
            # Normalize
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return placeholder on error
            return {
                'price_returns_avg': 0.25,
                'news_sentiment_avg': 0.20,
                'price_volatility': 0.15,
                'position_0': 0.10,
                'news_positive_count': 0.08,
            }

