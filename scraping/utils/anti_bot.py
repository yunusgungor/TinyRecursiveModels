"""
Anti-Bot Strategies
Implements various techniques to avoid bot detection
"""

import random
import logging
from typing import List, Dict, Any


class AntiBotHelper:
    """Helper class for anti-bot strategies"""
    
    def __init__(self, user_agents: List[str]):
        """
        Initialize anti-bot helper
        
        Args:
            user_agents: List of user agent strings to rotate
        """
        self.user_agents = user_agents
        self.logger = logging.getLogger(__name__)
        
        if not self.user_agents:
            # Default user agents if none provided
            self.user_agents = [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]
    
    def get_random_user_agent(self) -> str:
        """
        Get a random user agent string
        
        Returns:
            Random user agent string
        """
        user_agent = random.choice(self.user_agents)
        self.logger.debug(f"Selected user agent: {user_agent[:50]}...")
        return user_agent
    
    def get_browser_headers(self) -> Dict[str, str]:
        """
        Get realistic browser headers
        
        Returns:
            Dictionary of HTTP headers
        """
        return {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }

    def get_random_viewport(self) -> Dict[str, int]:
        """
        Get a random viewport size to simulate different devices
        
        Returns:
            Dictionary with width and height
        """
        viewports = [
            {'width': 1920, 'height': 1080},  # Full HD
            {'width': 1366, 'height': 768},   # Common laptop
            {'width': 1536, 'height': 864},   # HD+
            {'width': 1440, 'height': 900},   # MacBook
        ]
        return random.choice(viewports)
    
    async def simulate_human_behavior(self, page) -> None:
        """
        Simulate human-like behavior on a page
        
        Args:
            page: Playwright page object
        """
        try:
            # Random mouse movements
            await page.mouse.move(
                random.randint(100, 500),
                random.randint(100, 500)
            )
            
            # Random scroll
            scroll_amount = random.randint(100, 500)
            await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
            
            self.logger.debug("Simulated human behavior")
        except Exception as e:
            self.logger.warning(f"Could not simulate human behavior: {e}")
    
    def detect_captcha_keywords(self, page_content: str) -> bool:
        """
        Detect if page contains CAPTCHA keywords
        
        Args:
            page_content: HTML content of the page
            
        Returns:
            True if CAPTCHA detected, False otherwise
        """
        # More specific CAPTCHA detection patterns
        captcha_keywords = [
            'g-recaptcha',
            'recaptcha',
            'hcaptcha',
            'cf-challenge',
            'cloudflare',
            'verify you are human',
            'i am not a robot',
            'güvenlik doğrulaması',
            'robot değilim',
            'captcha-box'
        ]
        
        page_lower = page_content.lower()
        for keyword in captcha_keywords:
            if keyword in page_lower:
                self.logger.warning(f"CAPTCHA keyword detected: {keyword}")
                return True
        
        return False
