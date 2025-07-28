from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os

# Configuración de seguridad
SECRET_KEY = "NativoxNG2025SecretKey"  # En producción, usar una clave secreta más segura
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Configuración de autenticación
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__ident="2b")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Usuario predefinido
ADMIN_USER = {
    "username": "administrador@nativox.lat",
    "hashed_password": pwd_context.hash("NativoxNG2025"),
    "disabled": False
}

def verify_password(plain_password, hashed_password):
    """Verifica si la contraseña es correcta"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Genera un hash de la contraseña"""
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    """Autentica al usuario"""
    if username != ADMIN_USER["username"]:
        return False
    if not verify_password(password, ADMIN_USER["hashed_password"]):
        return False
    return ADMIN_USER

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crea un token de acceso JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Obtiene el usuario actual a partir del token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciales inválidas",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    if username != ADMIN_USER["username"]:
        raise credentials_exception
    
    return ADMIN_USER

def get_current_active_user(current_user = Depends(get_current_user)):
    """Verifica si el usuario está activo"""
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Usuario inactivo")
    return current_user 