"""
Firebase Firestore client for real-time state management and data persistence.
Handles all database operations with proper error handling and retry logic.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from google.api_core import exceptions, retry
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from config.settings import config

class FirebaseClient:
    """Firebase Firestore client with connection pooling and error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db = None
        self._initialize_firebase()
        
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            if not config.firebase_config:
                self.logger.warning("Firebase config not available. Running in offline mode.")
                return
                
            # Check if Firebase app is already initialized
            if not firebase_admin._apps:
                creds_dict = {
                    "type": "service_account",
                    "project_id": config.firebase_config.project_id,
                    "private_key_id": config.firebase_config.private_key_id,
                    "private_key": config.firebase_config.private_key,
                    "client_email": config.firebase_config.client_email,
                    "client_id": config.firebase_config.client_id,
                    "auth_uri": config.firebase_config.auth_uri,
                    "token_uri": config.firebase_config.token_uri,
                    "auth_provider_x509_cert_url": config.firebase_config.auth_provider_x509_cert_url,
                    "client_x509_cert_url": config.firebase_config.client_x509_cert_url
                }
                
                cred = credentials.Certificate(creds_dict)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            self.logger.info("Firebase Firestore initialized successfully")
            
        except ValueError as e:
            self.logger.error(f"Firebase app already exists: {e}")
            self.db = firestore.client()
        except FirebaseError as e:
            self.logger.error(f"Firebase initialization failed: {e}")
            self.db = None
        except Exception as e:
            self.logger.error