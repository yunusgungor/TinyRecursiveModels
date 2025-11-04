#!/usr/bin/env python3
"""Example script demonstrating Gmail API authentication."""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from auth import AuthenticationHandler, AuthConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate Gmail API authentication."""
    
    # Configuration
    config = AuthConfig(
        credentials_file="credentials.json",  # Download from Google Cloud Console
        token_file="token.json",
        scopes=['https://www.googleapis.com/auth/gmail.readonly'],
        use_encryption=True,  # Enable secure token storage
        encryption_password=None  # Will prompt user for password
    )
    
    # Initialize authentication handler
    auth_handler = AuthenticationHandler(config)
    
    print("Gmail Dataset Creator - Authentication Example")
    print("=" * 50)
    
    # Check current authentication state
    auth_state = auth_handler.get_auth_state()
    print(f"Current authentication state:")
    for key, value in auth_state.items():
        print(f"  {key}: {value}")
    print()
    
    # Attempt authentication
    print("Attempting authentication...")
    if auth_handler.authenticate():
        print("✅ Authentication successful!")
        
        # Get credentials
        credentials = auth_handler.get_credentials()
        if credentials:
            print(f"✅ Valid credentials obtained")
            print(f"   Scopes: {credentials.scopes}")
            print(f"   Valid: {credentials.valid}")
            print(f"   Has refresh token: {bool(credentials.refresh_token)}")
        
    else:
        print("❌ Authentication failed!")
        return 1
    
    # Test token refresh (if needed)
    print("\nTesting token refresh...")
    if auth_handler.refresh_token():
        print("✅ Token refresh successful")
    else:
        print("ℹ️  Token refresh not needed or failed")
    
    # Show final authentication state
    print("\nFinal authentication state:")
    final_state = auth_handler.get_auth_state()
    for key, value in final_state.items():
        print(f"  {key}: {value}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())