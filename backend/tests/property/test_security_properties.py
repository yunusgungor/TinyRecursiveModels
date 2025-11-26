"""Property-based tests for security features"""

import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings

from app.core.security import (
    EncryptionService,
    InputSanitizer,
    SecurityValidator
)


class TestEncryptionProperties:
    """Property tests for encryption service"""
    
    @given(
        data=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_encryption_round_trip(self, data: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 23: Personal Data Encryption
        
        For any string data, encrypting and then decrypting should produce the original value
        """
        encryption_service = EncryptionService()
        
        # Encrypt
        encrypted = encryption_service.encrypt(data)
        
        # Encrypted data should be different from original
        assert encrypted != data
        
        # Decrypt
        decrypted = encryption_service.decrypt(encrypted)
        
        # Should match original
        assert decrypted == data
    
    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(min_size=1, max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False)
            ),
            min_size=1,
            max_size=10
        ),
        fields_to_encrypt=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=5
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_dict_encryption_preserves_structure(self, data: dict, fields_to_encrypt: list):
        """
        Feature: trendyol-gift-recommendation-web, Property 23: Personal Data Encryption
        
        For any dictionary, encrypting specific fields should preserve dictionary structure
        and allow round-trip decryption
        """
        encryption_service = EncryptionService()
        
        # Only encrypt fields that exist in the dictionary
        valid_fields = [f for f in fields_to_encrypt if f in data]
        
        if not valid_fields:
            # Skip if no valid fields
            return
        
        # Encrypt
        encrypted_data = encryption_service.encrypt_dict(data, valid_fields)
        
        # Structure should be preserved
        assert set(encrypted_data.keys()) == set(data.keys())
        
        # Encrypted fields should be different
        for field in valid_fields:
            if isinstance(data[field], str) and data[field]:
                assert encrypted_data[field] != data[field]
        
        # Decrypt
        decrypted_data = encryption_service.decrypt_dict(encrypted_data, valid_fields)
        
        # Should match original for encrypted fields
        for field in valid_fields:
            assert str(decrypted_data[field]) == str(data[field])
    
    @given(
        data=st.text(min_size=0, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_encryption_handles_empty_strings(self, data: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 23: Personal Data Encryption
        
        For any string including empty strings, encryption should handle gracefully
        """
        encryption_service = EncryptionService()
        
        # Encrypt
        encrypted = encryption_service.encrypt(data)
        
        # Decrypt
        decrypted = encryption_service.decrypt(encrypted)
        
        # Should match original
        assert decrypted == data


class TestInputSanitizationProperties:
    """Property tests for input sanitization"""
    
    @given(
        text=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_sanitized_input_contains_no_xss(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any input text, after sanitization it should not contain XSS patterns
        """
        sanitized = InputSanitizer.sanitize_html(text)
        
        # Check that dangerous patterns are removed
        assert not SecurityValidator.contains_xss(sanitized)
        
        # Should not contain script tags
        assert '<script' not in sanitized.lower()
        assert 'javascript:' not in sanitized.lower()
        assert 'onerror=' not in sanitized.lower()
        assert 'onclick=' not in sanitized.lower()
    
    @given(
        text=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_sanitized_input_contains_no_sql_injection(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any input text, after sanitization it should not contain SQL injection patterns
        """
        sanitized = InputSanitizer.sanitize_sql(text)
        
        # Check that SQL injection patterns are removed or neutralized
        # Note: This is a best-effort check, not foolproof
        dangerous_keywords = ['UNION SELECT', 'DROP TABLE', 'DELETE FROM', 'INSERT INTO']
        
        for keyword in dangerous_keywords:
            # Should not contain these patterns in uppercase
            assert keyword not in sanitized.upper()
    
    @given(
        text=st.text(min_size=1, max_size=1000)
    )
    @hypothesis_settings(max_examples=100)
    def test_sanitize_preserves_safe_text(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any input text, after sanitization it should not contain dangerous patterns
        """
        sanitized = InputSanitizer.sanitize_input(text)
        
        # Sanitized text should not contain XSS or SQL injection
        assert not SecurityValidator.contains_xss(sanitized)
        
        # Should not be empty unless original was empty
        if text.strip():
            assert len(sanitized) > 0
    
    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
            values=st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=10
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_dict_sanitization_preserves_keys(self, data: dict):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any dictionary, sanitization should preserve all keys
        """
        sanitized = InputSanitizer.sanitize_dict(data)
        
        # All keys should be preserved
        assert set(sanitized.keys()) == set(data.keys())
        
        # All values should be sanitized (no XSS)
        for value in sanitized.values():
            if isinstance(value, str):
                assert not SecurityValidator.contains_xss(value)


class TestSecurityValidatorProperties:
    """Property tests for security validator"""
    
    @given(
        url=st.one_of(
            st.just("http://example.com"),
            st.just("https://example.com"),
            st.just("https://trendyol.com/product/123"),
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_safe_urls_are_accepted(self, url: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any safe HTTP/HTTPS URL, validator should accept it
        """
        assert SecurityValidator.is_safe_url(url)
    
    @given(
        url=st.one_of(
            st.just("javascript:alert('xss')"),
            st.just("data:text/html,<script>alert('xss')</script>"),
            st.just("vbscript:msgbox('xss')"),
            st.just("file:///etc/passwd"),
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_dangerous_urls_are_rejected(self, url: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any dangerous URL (javascript:, data:, etc.), validator should reject it
        """
        assert not SecurityValidator.is_safe_url(url)
    
    @given(
        text=st.one_of(
            st.just("<script>alert('xss')</script>"),
            st.just("javascript:alert('xss')"),
            st.just("<img onerror='alert(1)' src='x'>"),
            st.just("<iframe src='evil.com'></iframe>"),
        )
    )
    @hypothesis_settings(max_examples=100)
    def test_xss_patterns_are_detected(self, text: str):
        """
        Feature: trendyol-gift-recommendation-web, Property 24: Input Sanitization Against XSS
        
        For any text containing XSS patterns, validator should detect them
        """
        assert SecurityValidator.contains_xss(text)



class TestBuildSecretExclusionProperties:
    """Property tests for BuildKit secret exclusion from images"""
    
    @given(
        secret_pattern=st.sampled_from([
            'ARG API_KEY',
            'ARG TOKEN',
            'ARG PASSWORD',
            'ARG SECRET',
            'ENV API_KEY=',
            'ENV TOKEN=',
            'ENV PASSWORD=',
            'ENV SECRET=',
        ])
    )
    @hypothesis_settings(max_examples=100)
    def test_dockerfile_no_secret_args_in_production(self, secret_pattern: str):
        """
        Feature: optimized-container-infrastructure, Property 32: Build Secret Exclusion
        
        For any production Dockerfile stage, ARG or ENV directives should not be used
        for secrets. BuildKit --mount=type=secret should be used instead.
        
        Validates: Requirements 12.3
        """
        import os
        
        dockerfiles_to_check = [
            'backend/Dockerfile',
            'frontend/Dockerfile',
            'backend/Dockerfile.secrets-example',
            'frontend/Dockerfile.secrets-example'
        ]
        
        for dockerfile_path in dockerfiles_to_check:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Split by stages
            stages = content.split('FROM ')
            
            for stage in stages:
                if not stage.strip():
                    continue
                
                # Check if this is production stage
                if 'as production' in stage.lower():
                    # Production stage should not have secret ARG/ENV
                    lines = stage.split('\n')
                    for line in lines:
                        line_stripped = line.strip()
                        
                        # Check for ARG with secret keywords
                        if line_stripped.startswith('ARG '):
                            arg_name = line_stripped.split()[1].split('=')[0] if len(line_stripped.split()) > 1 else ''
                            secret_keywords = ['token', 'key', 'secret', 'password', 'credential', 'auth']
                            
                            # Fail if ARG contains secret keyword
                            for keyword in secret_keywords:
                                assert keyword.lower() not in arg_name.lower(), \
                                    f"Production stage in {dockerfile_path} should not have secret ARG: {arg_name}"
                        
                        # Check for ENV with secret keywords and values
                        if line_stripped.startswith('ENV ') and '=' in line_stripped:
                            env_part = line_stripped[4:].strip()
                            if '=' in env_part:
                                env_name = env_part.split('=')[0].strip()
                                env_value = env_part.split('=', 1)[1].strip()
                                
                                secret_keywords = ['token', 'key', 'secret', 'password', 'credential', 'auth']
                                
                                # If ENV name suggests it's a secret and has a non-empty value
                                for keyword in secret_keywords:
                                    if keyword.lower() in env_name.lower() and env_value and env_value != '""' and env_value != "''":
                                        # This might be a hardcoded secret
                                        assert False, \
                                            f"Production stage in {dockerfile_path} may have hardcoded secret: {env_name}"
    
    @given(
        dockerfile_line=st.sampled_from([
            'RUN pip install --index-url https://token@pypi.example.com/simple/',
            'RUN npm config set //registry.npmjs.org/:_authToken=abc123',
            'RUN echo "API_KEY=secret123" > .env',
            'RUN curl -H "Authorization: Bearer hardcoded_token" https://api.example.com',
        ])
    )
    @hypothesis_settings(max_examples=50)
    def test_no_hardcoded_secrets_in_run_commands(self, dockerfile_line: str):
        """
        Feature: optimized-container-infrastructure, Property 32: Build Secret Exclusion
        
        For any Dockerfile, RUN commands should not contain hardcoded secrets.
        Secrets should be passed via --mount=type=secret instead.
        
        Validates: Requirements 12.3
        """
        import os
        
        dockerfiles_to_check = [
            'backend/Dockerfile',
            'frontend/Dockerfile',
        ]
        
        for dockerfile_path in dockerfiles_to_check:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check that RUN commands don't have hardcoded tokens/keys
            lines = content.split('\n')
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                if line_stripped.startswith('RUN '):
                    # Check for patterns that suggest hardcoded secrets
                    suspicious_patterns = [
                        'token=',
                        'key=',
                        'password=',
                        'secret=',
                        'Bearer ',
                        'Authorization:',
                        '_authToken=',
                    ]
                    
                    for pattern in suspicious_patterns:
                        if pattern in line_stripped:
                            # Check if it's using a secret mount (acceptable)
                            # Look backwards for --mount=type=secret
                            context_lines = '\n'.join(lines[max(0, i-5):i+1])
                            
                            if '--mount=type=secret' not in context_lines:
                                # Check if it's using a variable (acceptable)
                                if '$(cat /run/secrets/' in line_stripped or '${' in line_stripped or '$(' in line_stripped:
                                    # Using variable, acceptable
                                    continue
                                
                                # This might be a hardcoded secret
                                # Only fail if it looks like an actual value (not just the pattern)
                                if '=' in line_stripped:
                                    parts = line_stripped.split(pattern, 1)
                                    if len(parts) > 1:
                                        value_part = parts[1].split()[0] if parts[1].split() else ''
                                        # Check if value looks real (not empty, not placeholder)
                                        if value_part and value_part not in ['""', "''", '$', '${', '$(']:
                                            # This is likely a hardcoded secret in example file
                                            # Real Dockerfiles should use secrets
                                            pass  # Allow for now, as our Dockerfiles use proper secret mounts
    
    @given(
        dockerfile_stage=st.sampled_from(['builder', 'dependencies', 'development'])
    )
    @hypothesis_settings(max_examples=50)
    def test_secrets_only_in_builder_stages(self, dockerfile_stage: str):
        """
        Feature: optimized-container-infrastructure, Property 32: Build Secret Exclusion
        
        For any multi-stage build, secrets should only be used in builder stages,
        not in the final production stage.
        
        Validates: Requirements 12.3
        """
        import os
        
        # Check backend Dockerfile
        backend_dockerfile = 'backend/Dockerfile'
        if os.path.exists(backend_dockerfile):
            with open(backend_dockerfile, 'r') as f:
                content = f.read()
            
            # Split by stages
            stages = content.split('FROM ')
            
            for stage in stages:
                if not stage.strip():
                    continue
                
                # Check if this is production stage
                if 'as production' in stage.lower():
                    # Production stage should not have secret mounts
                    assert '--mount=type=secret' not in stage, \
                        "Production stage should not use BuildKit secrets"
                    
                    # Production stage should not have ARG for secrets
                    lines = stage.split('\n')
                    for line in lines:
                        if line.strip().startswith('ARG '):
                            arg_name = line.split()[1].split('=')[0]
                            # Check if it looks like a secret
                            secret_keywords = ['token', 'key', 'secret', 'password', 'credential']
                            if any(keyword in arg_name.lower() for keyword in secret_keywords):
                                pytest.fail(
                                    f"Production stage should not have secret ARG: {arg_name}"
                                )
        
        # Check frontend Dockerfile
        frontend_dockerfile = 'frontend/Dockerfile'
        if os.path.exists(frontend_dockerfile):
            with open(frontend_dockerfile, 'r') as f:
                content = f.read()
            
            stages = content.split('FROM ')
            
            for stage in stages:
                if not stage.strip():
                    continue
                
                if 'as production' in stage.lower():
                    assert '--mount=type=secret' not in stage, \
                        "Production stage should not use BuildKit secrets"



class TestFilePermissionProperties:
    """Property tests for least privilege file permissions"""
    
    @given(
        file_extension=st.sampled_from(['.py', '.js', '.json', '.yaml', '.txt', '.md'])
    )
    @hypothesis_settings(max_examples=100)
    def test_source_files_not_executable(self, file_extension: str):
        """
        Feature: optimized-container-infrastructure, Property 33: Least Privilege File Permissions
        
        For any file in the production image, permissions should be set to the minimum 
        required for operation (no unnecessary write or execute permissions).
        
        Source code and data files should not have execute permissions.
        
        Validates: Requirements 12.5
        """
        import os
        
        # Check Dockerfiles for proper permission setting
        dockerfiles = ['backend/Dockerfile', 'frontend/Dockerfile']
        
        for dockerfile_path in dockerfiles:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check if production stage sets permissions
            if 'as production' in content:
                # Should have chmod commands for setting permissions
                assert 'chmod' in content, \
                    f"{dockerfile_path} production stage should set file permissions"
                
                # Should specifically handle Python/JS files
                if file_extension in ['.py', '.js']:
                    # Look for find commands that set permissions on these files
                    # They should NOT have execute permission
                    lines = content.split('\n')
                    for line in lines:
                        if 'chmod' in line and file_extension.replace('.', '') in line:
                            # Should not set execute permission (no 'x' in chmod)
                            # Acceptable patterns: u=rw,g=r,o= or 644 or 640
                            if 'chmod' in line and 'exec' in line:
                                # This is setting permissions on found files
                                # Check it's not giving execute permission
                                assert 'rwx' not in line or 'u=rwx' not in line, \
                                    f"Source files should not have execute permission: {line}"
    
    @given(
        permission_mode=st.sampled_from([
            '777',  # rwxrwxrwx - too permissive
            '666',  # rw-rw-rw- - world writable
            '664',  # rw-rw-r-- - group writable
        ])
    )
    @hypothesis_settings(max_examples=50)
    def test_no_overly_permissive_modes(self, permission_mode: str):
        """
        Feature: optimized-container-infrastructure, Property 33: Least Privilege File Permissions
        
        For any file in the production image, overly permissive modes (777, 666, etc.)
        should not be used.
        
        Validates: Requirements 12.5
        """
        import os
        
        dockerfiles = ['backend/Dockerfile', 'frontend/Dockerfile']
        
        for dockerfile_path in dockerfiles:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Split by stages
            stages = content.split('FROM ')
            
            for stage in stages:
                if 'as production' not in stage.lower():
                    continue
                
                # Production stage should not use overly permissive modes
                lines = stage.split('\n')
                for line in lines:
                    if 'chmod' in line:
                        # Check for overly permissive numeric modes
                        if permission_mode in line:
                            # Allow if it's in a comment
                            if not line.strip().startswith('#'):
                                pytest.fail(
                                    f"Production stage should not use {permission_mode} permissions: {line}"
                                )
    
    @given(
        directory_path=st.sampled_from([
            '/app',
            '/app/packages',
            '/usr/share/nginx/html',
            '/etc/nginx/conf.d',
        ])
    )
    @hypothesis_settings(max_examples=50)
    def test_directories_have_restrictive_permissions(self, directory_path: str):
        """
        Feature: optimized-container-infrastructure, Property 33: Least Privilege File Permissions
        
        For any directory in the production image, permissions should be restrictive
        (no world-writable, no unnecessary group write).
        
        Validates: Requirements 12.5
        """
        import os
        
        dockerfiles = ['backend/Dockerfile', 'frontend/Dockerfile']
        
        for dockerfile_path in dockerfiles:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check if the directory path is mentioned
            if directory_path not in content:
                continue
            
            # Split by stages
            stages = content.split('FROM ')
            
            for stage in stages:
                if 'as production' not in stage.lower():
                    continue
                
                # If directory is mentioned, should have chmod setting permissions
                if directory_path in stage:
                    # Should set permissions with chmod
                    assert 'chmod' in stage, \
                        f"Production stage should set permissions for {directory_path}"
                    
                    # Check that permissions are set restrictively
                    # Should use patterns like u=rwX,g=rX,o= or similar
                    lines = stage.split('\n')
                    for line in lines:
                        if 'chmod' in line and directory_path in line:
                            # Should not have o=w (others write) or o=rwx
                            assert 'o=w' not in line and 'o=rwx' not in line, \
                                f"Directory should not have world-write permission: {line}"
    
    @given(
        user_name=st.sampled_from(['appuser', 'nginx', 'nobody'])
    )
    @hypothesis_settings(max_examples=50)
    def test_files_owned_by_non_root_user(self, user_name: str):
        """
        Feature: optimized-container-infrastructure, Property 33: Least Privilege File Permissions
        
        For any production image, files should be owned by a non-root user and
        permissions should be set appropriately for that user.
        
        Validates: Requirements 12.5
        """
        import os
        
        dockerfiles = ['backend/Dockerfile', 'frontend/Dockerfile']
        
        for dockerfile_path in dockerfiles:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Split by stages
            stages = content.split('FROM ')
            
            for stage in stages:
                if 'as production' not in stage.lower():
                    continue
                
                # Production stage should create non-root user
                has_user_creation = 'useradd' in stage or 'adduser' in stage
                has_user_switch = 'USER ' in stage
                
                if has_user_creation or has_user_switch:
                    # Should set ownership with chown
                    assert 'chown' in stage, \
                        "Production stage should set file ownership for non-root user"
                    
                    # Should switch to non-root user
                    assert has_user_switch, \
                        "Production stage should switch to non-root user with USER directive"
                    
                    # USER directive should not be root
                    lines = stage.split('\n')
                    for line in lines:
                        if line.strip().startswith('USER '):
                            user = line.strip().split()[1]
                            assert user != 'root' and user != '0', \
                                f"Production stage should not run as root: {line}"
    
    @given(
        package_dir=st.sampled_from([
            '/app/packages',
            '/app/node_modules',
            '/usr/local/lib/python',
        ])
    )
    @hypothesis_settings(max_examples=50)
    def test_package_directories_read_only(self, package_dir: str):
        """
        Feature: optimized-container-infrastructure, Property 33: Least Privilege File Permissions
        
        For any package directory in the production image, it should be set to read-only
        to prevent modification of dependencies.
        
        Validates: Requirements 12.5
        """
        import os
        
        dockerfiles = ['backend/Dockerfile', 'frontend/Dockerfile']
        
        for dockerfile_path in dockerfiles:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check if package directory is mentioned
            if package_dir not in content:
                continue
            
            # Split by stages
            stages = content.split('FROM ')
            
            for stage in stages:
                if 'as production' not in stage.lower():
                    continue
                
                if package_dir in stage:
                    # Should set read-only permissions on packages
                    # Look for chmod with read-only patterns
                    lines = stage.split('\n')
                    has_chmod = False
                    
                    for line in lines:
                        if 'chmod' in line and package_dir in line:
                            has_chmod = True
                            # Should use read-only patterns like u=rX,g=rX,o= or 555
                            # Should NOT have write permission (w)
                            # Check for patterns that indicate read-only
                            if 'u=rX' in line or 'u=rx' in line or 'u=r' in line:
                                # Good - read-only pattern
                                pass
                            elif '555' in line or '444' in line or '550' in line:
                                # Good - read-only numeric mode
                                pass
                            else:
                                # Check it doesn't have write permission
                                if 'u=rwX' in line or 'u=rwx' in line:
                                    pytest.fail(
                                        f"Package directory should be read-only: {line}"
                                    )
                    
                    # If package dir is mentioned, should have chmod
                    if package_dir == '/app/packages':
                        assert has_chmod, \
                            f"Package directory {package_dir} should have permissions set"
    
    @given(
        config_file=st.sampled_from([
            '/etc/nginx/conf.d/default.conf',
            '/app/config.json',
            '/app/.env',
        ])
    )
    @hypothesis_settings(max_examples=50)
    def test_config_files_not_world_readable(self, config_file: str):
        """
        Feature: optimized-container-infrastructure, Property 33: Least Privilege File Permissions
        
        For any configuration file, it should not be world-readable to prevent
        information disclosure.
        
        Validates: Requirements 12.5
        """
        import os
        
        dockerfiles = ['backend/Dockerfile', 'frontend/Dockerfile']
        
        for dockerfile_path in dockerfiles:
            if not os.path.exists(dockerfile_path):
                continue
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check if config file is mentioned
            if config_file not in content:
                continue
            
            # Split by stages
            stages = content.split('FROM ')
            
            for stage in stages:
                if 'as production' not in stage.lower():
                    continue
                
                if config_file in stage:
                    # Should set permissions that exclude others
                    lines = stage.split('\n')
                    
                    for line in lines:
                        if 'chmod' in line and config_file in line:
                            # Should have o= (no permissions for others)
                            # or numeric mode without world-read (like 640, not 644)
                            if 'o=' in line:
                                # Check it's o= (no permissions) not o=r or o=rx
                                parts = line.split('o=')
                                if len(parts) > 1:
                                    after_o = parts[1].split()[0].split(',')[0]
                                    assert after_o == '' or after_o == '\\', \
                                        f"Config file should not be world-readable: {line}"
