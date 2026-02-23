import jwt
import bcrypt
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import re
from pathlib import Path
import os
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role levels."""
    ADMIN = "admin"
    TRAINER = "trainer"
    VIEWER = "viewer"

class TokenType(Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    API = "api"

class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class User:
    """User data model."""
    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    api_key_hash: Optional[str] = None
    api_key_created_at: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

@dataclass
class SecurityAuditLog:
    """Security audit log entry."""
    id: str
    user_id: str
    action: str
    resource: str
    security_level: SecurityLevel
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

class SecurityConfig:
    """Security configuration."""
    
    def __init__(self):
        # JWT settings
        self.jwt_secret_key = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.api_key_expire_days = int(os.environ.get("API_KEY_EXPIRE_DAYS", "365"))
        
        # Password settings
        self.password_min_length = int(os.environ.get("PASSWORD_MIN_LENGTH", "8"))
        self.password_require_uppercase = os.environ.get("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true"
        self.password_require_lowercase = os.environ.get("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true"
        self.password_require_numbers = os.environ.get("PASSWORD_REQUIRE_NUMBERS", "true").lower() == "true"
        self.password_require_special = os.environ.get("PASSWORD_REQUIRE_SPECIAL", "true").lower() == "true"
        
        # Security settings
        self.max_failed_login_attempts = int(os.environ.get("MAX_FAILED_LOGIN_ATTEMPTS", "5"))
        self.account_lockout_duration_minutes = int(os.environ.get("ACCOUNT_LOCKOUT_DURATION_MINUTES", "30"))
        self.session_timeout_minutes = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "60"))
        self.enable_rate_limiting = os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true"
        self.rate_limit_requests_per_minute = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
        
        # Encryption settings
        self.encryption_key = os.environ.get("ENCRYPTION_KEY", secrets.token_urlsafe(32))
        
        # CORS settings
        self.allowed_origins = json.loads(os.environ.get("ALLOWED_ORIGINS", '["http://localhost:3000"]'))
        self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = ["*"]
        
        # File upload settings
        self.max_file_size_mb = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))
        self.allowed_file_extensions = json.loads(os.environ.get("ALLOWED_FILE_EXTENSIONS", '[".json", ".jsonl", ".csv", ".txt", ".parquet", ".xlsx"]'))
        
        # API settings
        self.api_rate_limit_per_hour = int(os.environ.get("API_RATE_LIMIT_PER_HOUR", "1000"))
        self.api_key_length = int(os.environ.get("API_KEY_LENGTH", "32"))

class PasswordValidator:
    """Password validation and security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against security requirements."""
        errors = []
        
        # Length check
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        # Character requirements
        if self.config.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Common password check
        common_passwords = [
            "password", "123456", "12345678", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey"
        ]
        
        if password.lower() in common_passwords:
            errors.append("Password is too common, please choose a more secure password")
        
        # Sequential characters check
        if re.search(r'(.)\1{2,}', password):
            errors.append("Password contains too many repeated characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "strength": self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> str:
        """Calculate password strength score."""
        score = 0
        
        # Length score
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        
        # Character variety score
        if re.search(r'[a-z]', password):
            score += 1
        if re.search(r'[A-Z]', password):
            score += 1
        if re.search(r'\d', password):
            score += 1
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        
        # Complexity bonus
        if len(set(password)) >= len(password) * 0.7:  # Good character variety
            score += 1
        
        # Determine strength
        if score >= 6:
            return "very_strong"
        elif score >= 4:
            return "strong"
        elif score >= 2:
            return "medium"
        else:
            return "weak"
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False

class JWTManager:
    """JWT token management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)
        
        expire = datetime.now(timezone.utc) + expires_delta
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "type": TokenType.ACCESS.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.now(timezone.utc) + timedelta(days=self.config.refresh_token_expire_days)
        
        payload = {
            "sub": user.id,
            "type": TokenType.REFRESH.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16)
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def create_api_key(self, user: User) -> str:
        """Create API key for programmatic access."""
        # Generate secure API key
        api_key = f"qlora_{secrets.token_urlsafe(self.config.api_key_length)}"
        
        # Create API key hash for storage
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Create API key token
        expire = datetime.now(timezone.utc) + timedelta(days=self.config.api_key_expire_days)
        
        payload = {
            "sub": user.id,
            "type": TokenType.API.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": api_key_hash[:16]
        }
        
        api_token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        
        return {
            "api_key": api_key,
            "api_token": api_token,
            "api_key_hash": api_key_hash,
            "expires_at": expire
        }
    
    def verify_token(self, token: str, token_type: Optional[TokenType] = None) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            
            # Check token type if specified
            if token_type and payload.get("type") != token_type.value:
                raise jwt.InvalidTokenError(f"Invalid token type. Expected {token_type.value}, got {payload.get('type')}")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
                raise jwt.ExpiredSignatureError("Token has expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise jwt.InvalidTokenError("Invalid token")
    
    def revoke_token(self, token: str) -> bool:
        """Revoke JWT token."""
        try:
            # In a real implementation, you would store revoked tokens in a database
            # For now, we'll just verify the token exists
            self.verify_token(token)
            logger.info("Token revoked (implementation needed for persistent revocation)")
            return True
        except Exception as e:
            logger.error(f"Error revoking token: {str(e)}")
            return False

class SecurityAudit:
    """Security audit logging."""
    
    def __init__(self, db=None):
        self.db = db
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def log_security_event(self, user_id: str, action: str, resource: str,
                                security_level: SecurityLevel, ip_address: str,
                                user_agent: str, success: bool, 
                                error_message: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """Log security event."""
        try:
            audit_log = SecurityAuditLog(
                id=f"audit_{secrets.token_urlsafe(16)}",
                user_id=user_id,
                action=action,
                resource=resource,
                security_level=security_level,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                error_message=error_message,
                metadata=metadata or {},
                timestamp=datetime.now()
            )
            
            # Store in database
            if self.db:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self._store_audit_log_sync, 
                    audit_log
                )
            
            # Log security event
            if success:
                logger.info(f"Security event: {user_id} {action} {resource} ({security_level.value})")
            else:
                logger.warning(f"Security event failed: {user_id} {action} {resource} ({security_level.value}) - {error_message}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {str(e)}")
    
    def _store_audit_log_sync(self, audit_log: SecurityAuditLog):
        """Store audit log synchronously."""
        try:
            audit_dict = asdict(audit_log)
            audit_dict["timestamp"] = audit_log.timestamp.isoformat()
            self.db.security_audit_logs.insert_one(audit_dict)
        except Exception as e:
            logger.error(f"Error storing audit log: {str(e)}")
    
    async def get_audit_logs(self, user_id: Optional[str] = None, 
                           action: Optional[str] = None,
                           security_level: Optional[SecurityLevel] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs with filters."""
        try:
            query = {}
            if user_id:
                query["user_id"] = user_id
            if action:
                query["action"] = action
            if security_level:
                query["security_level"] = security_level.value
            
            if self.db:
                logs = list(self.db.security_audit_logs.find(query).sort("timestamp", -1).limit(limit))
                
                # Convert ObjectId and datetime to strings
                for log in logs:
                    log["_id"] = str(log["_id"])
                    if "timestamp" in log and hasattr(log["timestamp"], 'isoformat'):
                        log["timestamp"] = log["timestamp"].isoformat()
                
                return logs
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting audit logs: {str(e)}")
            return []
    
    async def detect_suspicious_activity(self, user_id: str, 
                                        time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Detect suspicious activity for a user."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            if not self.db:
                return []
            
            # Get recent failed login attempts
            failed_logins = list(self.db.security_audit_logs.find({
                "user_id": user_id,
                "action": "login",
                "success": False,
                "timestamp": {"$gte": cutoff_time}
            }).sort("timestamp", 1))
            
            suspicious_activities = []
            
            # Check for multiple failed login attempts
            if len(failed_logins) >= self.config.max_failed_login_attempts:
                suspicious_activities.append({
                    "type": "multiple_failed_logins",
                    "severity": "high",
                    "description": f"{len(failed_logins)} failed login attempts in {time_window_minutes} minutes",
                    "count": len(failed_logins),
                    "first_attempt": failed_logins[0]["timestamp"].isoformat() if failed_logins else None,
                    "last_attempt": failed_logins[-1]["timestamp"].isoformat() if failed_logins else None
                })
            
            # Check for rapid API requests (potential brute force)
            rapid_api_requests = list(self.db.security_audit_logs.find({
                "user_id": user_id,
                "action": {"$in": ["api_request", "api_key_usage"]},
                "timestamp": {"$gte": datetime.now() - timedelta(minutes=5)}
            }).sort("timestamp", 1))
            
            if len(rapid_api_requests) > 50:  # More than 50 API requests in 5 minutes
                suspicious_activities.append({
                    "type": "rapid_api_requests",
                    "severity": "medium",
                    "description": f"{len(rapid_api_requests)} API requests in 5 minutes",
                    "count": len(rapid_api_requests)
                })
            
            return suspicious_activities
            
        except Exception as e:
            logger.error(f"Error detecting suspicious activity: {str(e)}")
            return []

class EnhancedAuthManager:
    """Enhanced authentication manager with comprehensive security features."""
    
    def __init__(self, db=None):
        self.db = db
        self.config = SecurityConfig()
        self.jwt_manager = JWTManager(self.config)
        self.password_validator = PasswordValidator(self.config)
        self.security_audit = SecurityAudit(db)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage = {}
        self.active_sessions = {}
    
    async def register_user(self, username: str, email: str, password: str, 
                          role: UserRole = UserRole.VIEWER, 
                          ip_address: str = "unknown",
                          user_agent: str = "unknown") -> Dict[str, Any]:
        """Register new user with security validation."""
        try:
            # Validate input
            if not self._validate_username(username):
                await self.security_audit.log_security_event(
                    "system", "user_registration", username,
                    SecurityLevel.MEDIUM, ip_address, user_agent, False,
                    "Invalid username format"
                )
                return {"success": False, "error": "Invalid username format"}
            
            if not self._validate_email(email):
                await self.security_audit.log_security_event(
                    "system", "user_registration", username,
                    SecurityLevel.MEDIUM, ip_address, user_agent, False,
                    "Invalid email format"
                )
                return {"success": False, "error": "Invalid email format"}
            
            # Validate password
            password_validation = self.password_validator.validate_password(password)
            if not password_validation["valid"]:
                await self.security_audit.log_security_event(
                    "system", "user_registration", username,
                    SecurityLevel.MEDIUM, ip_address, user_agent, False,
                    f"Password validation failed: {', '.join(password_validation['errors'])}"
                )
                return {"success": False, "error": "Password validation failed", "details": password_validation["errors"]}
            
            # Check if user already exists
            existing_user = await self._get_user_by_username(username)
            if existing_user:
                await self.security_audit.log_security_event(
                    "system", "user_registration", username,
                    SecurityLevel.MEDIUM, ip_address, user_agent, False,
                    "Username already exists"
                )
                return {"success": False, "error": "Username already exists"}
            
            existing_email = await self._get_user_by_email(email)
            if existing_email:
                await self.security_audit.log_security_event(
                    "system", "user_registration", username,
                    SecurityLevel.MEDIUM, ip_address, user_agent, False,
                    "Email already exists"
                )
                return {"success": False, "error": "Email already exists"}
            
            # Create user
            user_id = f"user_{secrets.token_urlsafe(8)}"
            password_hash = self.password_validator.hash_password(password)
            
            user = User(
                id=user_id,
                username=username,
                email=email,
                role=role,
                is_active=True,
                created_at=datetime.now(),
                failed_login_attempts=0
            )
            
            # Store user in database
            user_dict = asdict(user)
            user_dict["password_hash"] = password_hash
            user_dict["role"] = user.role.value
            user_dict["created_at"] = user.created_at.isoformat()
            
            if self.db:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.db.users.insert_one, user_dict
                )
            
            # Log successful registration
            await self.security_audit.log_security_event(
                user_id, "user_registration", username,
                SecurityLevel.MEDIUM, ip_address, user_agent, True,
                metadata={"role": role.value}
            )
            
            logger.info(f"User {username} registered successfully with role {role.value}")
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "User registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            await self.security_audit.log_security_event(
                "system", "user_registration", username,
                SecurityLevel.HIGH, ip_address, user_agent, False,
                str(e)
            )
            return {"success": False, "error": "Registration failed"}
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = "unknown",
                              user_agent: str = "unknown") -> Dict[str, Any]:
        """Authenticate user with security checks."""
        try:
            # Get user
            user = await self._get_user_by_username(username)
            if not user:
                await self.security_audit.log_security_event(
                    username, "login", "authentication",
                    SecurityLevel.MEDIUM, ip_address, user_agent, False,
                    "User not found"
                )
                return {"success": False, "error": "Invalid credentials"}
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.now():
                await self.security_audit.log_security_event(
                    user.id, "login", "authentication",
                    SecurityLevel.HIGH, ip_address, user_agent, False,
                    "Account is locked"
                )
                return {"success": False, "error": "Account is locked. Please try again later."}
            
            # Verify password
            password_valid = self.password_validator.verify_password(password, user.password_hash)
            
            if not password_valid:
                # Increment failed login attempts
                user.failed_login_attempts += 1
                
                # Lock account if max attempts reached
                if user.failed_login_attempts >= self.config.max_failed_login_attempts:
                    user.locked_until = datetime.now() + timedelta(minutes=self.config.account_lockout_duration_minutes)
                    await self.security_audit.log_security_event(
                        user.id, "login", "authentication",
                        SecurityLevel.CRITICAL, ip_address, user_agent, False,
                        "Account locked due to too many failed attempts"
                    )
                
                # Update user in database
                if self.db:
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.db.users.update_one,
                        {"id": user.id},
                        {
                            "$set": {
                                "failed_login_attempts": user.failed_login_attempts,
                                "locked_until": user.locked_until.isoformat() if user.locked_until else None
                            }
                        }
                    )
                
                await self.security_audit.log_security_event(
                    user.id, "login", "authentication",
                    SecurityLevel.MEDIUM, ip_address, user_agent, False,
                    "Invalid password"
                )
                
                return {"success": False, "error": "Invalid credentials"}
            
            # Reset failed login attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            
            # Update user in database
            if self.db:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.db.users.update_one,
                    {"id": user.id},
                    {
                        "$set": {
                            "failed_login_attempts": 0,
                            "locked_until": None,
                            "last_login": user.last_login.isoformat()
                        }
                    }
                )
            
            # Create tokens
            access_token = self.jwt_manager.create_access_token(user)
            refresh_token = self.jwt_manager.create_refresh_token(user)
            
            # Log successful login
            await self.security_audit.log_security_event(
                user.id, "login", "authentication",
                SecurityLevel.MEDIUM, ip_address, user_agent, True
            )
            
            # Store session
            session_id = f"session_{secrets.token_urlsafe(16)}"
            self.active_sessions[session_id] = {
                "user_id": user.id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "ip_address": ip_address,
                "user_agent": user_agent
            }
            
            logger.info(f"User {username} authenticated successfully")
            
            return {
                "success": True,
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "session_id": session_id,
                "expires_in": self.config.access_token_expire_minutes * 60
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            await self.security_audit.log_security_event(
                username, "login", "authentication",
                SecurityLevel.HIGH, ip_address, user_agent, False,
                str(e)
            )
            return {"success": False, "error": "Authentication failed"}
    
    async def refresh_access_token(self, refresh_token: str, 
                                 ip_address: str = "unknown",
                                 user_agent: str = "unknown") -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        try:
            # Verify refresh token
            payload = self.jwt_manager.verify_token(refresh_token, TokenType.REFRESH)
            user_id = payload["sub"]
            
            # Get user
            user = await self._get_user_by_id(user_id)
            if not user or not user.is_active:
                await self.security_audit.log_security_event(
                    user_id, "token_refresh", "authentication",
                    SecurityLevel.HIGH, ip_address, user_agent, False,
                    "User not found or inactive"
                )
                return {"success": False, "error": "Invalid refresh token"}
            
            # Create new access token
            access_token = self.jwt_manager.create_access_token(user)
            
            await self.security_audit.log_security_event(
                user_id, "token_refresh", "authentication",
                SecurityLevel.LOW, ip_address, user_agent, True
            )
            
            return {
                "success": True,
                "access_token": access_token,
                "expires_in": self.config.access_token_expire_minutes * 60
            }
            
        except jwt.ExpiredSignatureError:
            await self.security_audit.log_security_event(
                "unknown", "token_refresh", "authentication",
                SecurityLevel.MEDIUM, ip_address, user_agent, False,
                "Refresh token expired"
            )
            return {"success": False, "error": "Refresh token expired"}
        except Exception as e:
            logger.error(f"Error refreshing token: {str(e)}")
            await self.security_audit.log_security_event(
                "unknown", "token_refresh", "authentication",
                SecurityLevel.MEDIUM, ip_address, user_agent, False,
                str(e)
            )
            return {"success": False, "error": "Token refresh failed"}
    
    async def create_api_key(self, user_id: str, 
                           ip_address: str = "unknown",
                           user_agent: str = "unknown") -> Dict[str, Any]:
        """Create API key for user."""
        try:
            # Get user
            user = await self._get_user_by_id(user_id)
            if not user or not user.is_active:
                await self.security_audit.log_security_event(
                    user_id, "api_key_creation", "authentication",
                    SecurityLevel.HIGH, ip_address, user_agent, False,
                    "User not found or inactive"
                )
                return {"success": False, "error": "User not found or inactive"}
            
            # Create API key
            api_key_data = self.jwt_manager.create_api_key(user)
            
            # Update user with API key hash
            if self.db:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.db.users.update_one,
                    {"id": user_id},
                    {
                        "$set": {
                            "api_key_hash": api_key_data["api_key_hash"],
                            "api_key_created_at": datetime.now().isoformat()
                        }
                    }
                )
            
            await self.security_audit.log_security_event(
                user_id, "api_key_creation", "authentication",
                SecurityLevel.MEDIUM, ip_address, user_agent, True
            )
            
            return {
                "success": True,
                "api_key": api_key_data["api_key"],
                "expires_at": api_key_data["expires_at"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating API key: {str(e)}")
            await self.security_audit.log_security_event(
                user_id, "api_key_creation", "authentication",
                SecurityLevel.HIGH, ip_address, user_agent, False,
                str(e)
            )
            return {"success": False, "error": "API key creation failed"}
    
    async def validate_api_key(self, api_key: str, 
                             ip_address: str = "unknown",
                             user_agent: str = "unknown") -> Dict[str, Any]:
        """Validate API key."""
        try:
            # Hash the provided API key
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Find user with matching API key hash
            if self.db:
                user_data = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.db.users.find_one,
                    {"api_key_hash": api_key_hash, "is_active": True}
                )
                
                if user_data:
                    user = self._user_from_dict(user_data)
                    
                    # Log API key usage
                    await self.security_audit.log_security_event(
                        user.id, "api_key_usage", "authentication",
                        SecurityLevel.LOW, ip_address, user_agent, True
                    )
                    
                    return {
                        "success": True,
                        "user_id": user.id,
                        "username": user.username,
                        "role": user.role.value
                    }
            
            await self.security_audit.log_security_event(
                "unknown", "api_key_usage", "authentication",
                SecurityLevel.MEDIUM, ip_address, user_agent, False,
                "Invalid API key"
            )
            
            return {"success": False, "error": "Invalid API key"}
            
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            await self.security_audit.log_security_event(
                "unknown", "api_key_usage", "authentication",
                SecurityLevel.MEDIUM, ip_address, user_agent, False,
                str(e)
            )
            return {"success": False, "error": "API key validation failed"}
    
    def verify_token(self, token: str, token_type: Optional[TokenType] = None) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = self.jwt_manager.verify_token(token, token_type)
            return {"success": True, "payload": payload}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def require_auth(self, required_roles: Optional[List[UserRole]] = None, 
                    security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """Decorator for requiring authentication."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # This would be implemented based on your web framework
                # For FastAPI, you'd use dependency injection
                # For now, this is a placeholder
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format."""
        if len(username) < 3 or len(username) > 30:
            return False
        
        # Allow alphanumeric characters, underscores, and hyphens
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            return False
        
        return True
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        if not self.db:
            return None
        
        user_data = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.db.users.find_one,
            {"username": username}
        )
        
        if user_data:
            return self._user_from_dict(user_data)
        
        return None
    
    async def _get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        if not self.db:
            return None
        
        user_data = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.db.users.find_one,
            {"email": email}
        )
        
        if user_data:
            return self._user_from_dict(user_data)
        
        return None
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        if not self.db:
            return None
        
        user_data = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.db.users.find_one,
            {"id": user_id}
        )
        
        if user_data:
            return self._user_from_dict(user_data)
        
        return None
    
    def _user_from_dict(self, user_data: Dict[str, Any]) -> User:
        """Convert user dictionary to User object."""
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            role=UserRole(user_data["role"]),
            is_active=user_data["is_active"],
            created_at=datetime.fromisoformat(user_data["created_at"]),
            last_login=datetime.fromisoformat(user_data["last_login"]) if user_data.get("last_login") else None,
            failed_login_attempts=user_data.get("failed_login_attempts", 0),
            locked_until=datetime.fromisoformat(user_data["locked_until"]) if user_data.get("locked_until") else None,
            api_key_hash=user_data.get("api_key_hash"),
            api_key_created_at=datetime.fromisoformat(user_data["api_key_created_at"]) if user_data.get("api_key_created_at") else None,
            mfa_enabled=user_data.get("mfa_enabled", False),
            mfa_secret=user_data.get("mfa_secret")
        )

# Global auth manager instance
_global_auth_manager = None

def initialize_auth_manager(db=None) -> EnhancedAuthManager:
    """Initialize global authentication manager."""
    global _global_auth_manager
    
    if _global_auth_manager is None:
        _global_auth_manager = EnhancedAuthManager(db)
        logger.info("Global authentication manager initialized")
    
    return _global_auth_manager

def get_auth_manager() -> EnhancedAuthManager:
    """Get the global authentication manager."""
    if _global_auth_manager is None:
        raise RuntimeError("Authentication manager not initialized. Call initialize_auth_manager() first.")
    
    return _global_auth_manager