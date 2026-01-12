import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None

class AuthManager:
    """
    Authentication dan authorization manager dengan proper security practices.
    """
    
    def __init__(self, secret_key: str = None, db=None):
        self.secret_key = secret_key or os.environ.get("SECRET_KEY", self._generate_secret_key())
        self.db = db
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
        # Generate encryption key untuk sensitive data
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _generate_secret_key(self) -> str:
        """Generate secure secret key."""
        return Fernet.generate_key().decode()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[int] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        # Set expiration
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + datetime.timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        # Create token
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str, username: str) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + datetime.timedelta(days=self.refresh_token_expire_days)
        
        data = {
            "sub": user_id,
            "username": username,
            "exp": expire,
            "type": "refresh"
        }
        
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Check expiration (jwt.decode already checks this)
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return TokenData(username=username, user_id=user_id)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user dengan username dan password."""
        if not self.db:
            return None
        
        try:
            # Find user in database
            user_doc = await self.db.users.find_one({"username": username})
            
            if not user_doc:
                return None
            
            # Verify password
            if not self.verify_password(password, user_doc["hashed_password"]):
                return None
            
            # Update last login
            await self.db.users.update_one(
                {"_id": user_doc["_id"]},
                {"$set": {"last_login": datetime.utcnow().isoformat()}}
            )
            
            # Return user object
            return User(
                id=str(user_doc["_id"]),
                username=user_doc["username"],
                email=user_doc["email"],
                is_active=user_doc.get("is_active", True),
                created_at=user_doc["created_at"],
                last_login=user_doc.get("last_login")
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None
    
    async def create_user(self, username: str, email: str, password: str) -> User:
        """Create new user dengan proper validation."""
        if not self.db:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database not available"
            )
        
        try:
            # Check if user already exists
            existing_user = await self.db.users.find_one({"username": username})
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already registered"
                )
            
            existing_email = await self.db.users.find_one({"email": email})
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Validate password strength
            if len(password) < 8:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Password must be at least 8 characters long"
                )
            
            # Hash password
            hashed_password = self.get_password_hash(password)
            
            # Create user document
            user_doc = {
                "username": username,
                "email": email,
                "hashed_password": hashed_password,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": None,
                "role": "user",  # Default role
                "permissions": ["read", "write"]  # Default permissions
            }
            
            # Insert into database
            result = await self.db.users.insert_one(user_doc)
            
            # Return user object
            return User(
                id=str(result.inserted_id),
                username=username,
                email=email,
                is_active=True,
                created_at=user_doc["created_at"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"User creation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def check_permissions(self, user: User, required_permissions: list) -> bool:
        """Check if user has required permissions."""
        if not user.is_active:
            return False
        
        # For now, implement basic permission checking
        # This can be expanded with more sophisticated RBAC
        user_permissions = ["read", "write"]  # Default permissions for active users
        
        return all(perm in user_permissions for perm in required_permissions)
    
    async def log_user_activity(self, user_id: str, activity: str, details: Dict[str, Any] = None):
        """Log user activity untuk audit trail."""
        if not self.db:
            return
        
        try:
            activity_log = {
                "user_id": user_id,
                "activity": activity,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": details.get("ip_address") if details else None,
                "user_agent": details.get("user_agent") if details else None
            }
            
            await self.db.user_activities.insert_one(activity_log)
            
        except Exception as e:
            logger.error(f"Error logging user activity: {str(e)}")

# Security utilities
def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token."""
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username: str) -> bool:
    """Validate username format."""
    import re
    pattern = r'^[a-zA-Z0-9_-]{3,20}$'
    return re.match(pattern, username) is not None

# Rate limiting decorator (simple implementation)
def rate_limit(max_calls: int = 100, time_window: int = 3600):
    """
    Simple rate limiting decorator.
    Note: For production, use Redis or similar for distributed rate limiting.
    """
    def decorator(func):
        call_history = []
        
        def wrapper(*args, **kwargs):
            import time
            current_time = time.time()
            
            # Remove old calls outside time window
            call_history[:] = [call_time for call_time in call_history if current_time - call_time < time_window]
            
            # Check if limit exceeded
            if len(call_history) >= max_calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            # Add current call
            call_history.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator