
# coding: utf-8

import re
class Tweet_process:
    """
    A helper class for processing tweets. Input: unprocessed tweets
    """
    emojis = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
        "]+", flags=re.UNICODE)
    hashtags = "(#[A-Za-z0-9_]+)"
    users = "(@[A-Za-z0-9_]+)"
    urls = "(https://t.co/[A-Za-z0-9]+)"
    
    def __init__(self, raw_tweet):
        self.raw_tweet = raw_tweet
    
    def tweet_cleaner(self):
        """
        Remove non-alphabetic characters including emojis, hashtags, tagged user addresss, url and repeated characters
        """
        emoji_remove = Tweet_process.emojis.sub(r'', self)
        patterns = Tweet_process.hashtags + "|" + Tweet_process.users + "|" + Tweet_process.urls + "|(&[A-Za-z0-9]+)"
        patterns_remove = re.sub(patterns,'', emoji_remove)
        characters_remove = re.sub(r'\W+', ' ', patterns_remove)
        cleaned = re.sub(r'\s+', ' ', characters_remove)
    
        return cleaned
        
    def get_emojis(self):
        """
        Find and return a list of all emojis from a unprocessed tweets
        """   
        emojis = re.findall(Tweet_process.emojis, self)
        return emojis
        
    def get_hashtags(self):
        """
        Find and return a list of all hashtags from a unprocessed tweets
        """
        hashtags = re.findall(Tweet_process.hashtags, self)
        return hashtags
        
    def get_users(self):
        """
        Find and return a list of all tagged user addresss from a unprocessed tweets
        """
        users = re.findall(Tweet_process.users, self)
        return users