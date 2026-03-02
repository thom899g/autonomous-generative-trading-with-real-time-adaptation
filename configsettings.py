"""
Configuration management for the Autonomous Generative Trading System.
Uses environment variables with Firebase as the primary configuration source.
"""
import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str = "https://accounts.google.com/o/oauth2/auth"
    token_uri: str = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url: str = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url: str = ""

@dataclass
class TradingConfig:
    """Trading system configuration"""
    max_position_size: float = 0.1  # 10% of portfolio per trade
    max_daily_loss: float = 0.02    # 2% max daily loss
    risk_free_rate: float = 0.02    # 2% annual risk-free rate
    commission_rate: float = 0.001  # 0.1% commission
    slippage_factor: float = 0.0005 # 0.05% slippage

@dataclass
class ModelConfig:
    """Model training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    sequence_length: int = 50
    hidden_dim: int = 256
    num_layers: int = 3

class ConfigManager:
    """Manages configuration with Firebase fallback"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.firebase_config = self._load_firebase_config()
        self.trading_config = TradingConfig()
        self.model_config = ModelConfig()
        
    def _load_firebase_config(self) -> Optional[FirebaseConfig]:
        """Load Firebase configuration from environment or file"""
        try:
            # Try environment variables first
            firebase_creds = os.getenv("FIREBASE_CREDENTIALS_JSON")
            
            if firebase_creds:
                # Parse JSON string from environment
                creds_dict = json.loads(firebase_creds)
            else:
                # Try file path
                creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
                if os.path.exists(creds_path):
                    with open(creds_path, 'r') as f:
                        creds_dict = json.load(f)
                else:
                    self.logger.warning("No Firebase credentials found. System will run in offline mode.")
                    return None
            
            return FirebaseConfig(
                project_id=creds_dict.get("project_id", ""),
                private_key_id=creds_dict.get("private_key_id", ""),
                private_key=creds_dict.get("private_key", "").replace('\\n', '\n'),
                client_email=creds_dict.get("client_email", ""),
                client_id=creds_dict.get("client_id", ""),
                client_x509_cert_url=creds_dict.get("client_x509_cert_url", "")
            )
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load Firebase config: {e}")
            return None
    
    def get_all_config(self) -> Dict[str, Any]:
        """Return complete configuration dictionary"""
        return {
            "firebase": self.firebase_config.__dict__ if self.firebase_config else None,
            "trading": self.trading_config.__dict__,
            "model": self.model_config.__dict__
        }

# Global configuration instance
config = ConfigManager()