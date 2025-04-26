from django.shortcuts import render
import re
import subprocess

def extract_tweet_id(url):
    """
    Extract the tweet ID from a Twitter URL.
    Example URL formats:
    - https://twitter.com/username/status/1234567890
    - https://x.com/username/status/1234567890
    """
    # Pattern to match Twitter URLs and extract the ID
    patterns = [
        r'twitter\.com/\w+/status/(\d+)',
        r'x\.com/\w+/status/(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def run_fake_news_detection(tweet_id):
    """
    Run the fake news detection script with the tweet ID as an argument.
    This function should call the AI model script that will be implemented by the user.
    
    For now, this is a placeholder that returns a dummy result.
    The actual implementation will involve calling a Python script with subprocess.
    """
    try:
        # In a real implementation, this would call the user's AI model script
        # Example: result = subprocess.check_output(['python', 'fake_news_model.py', tweet_id], text=True).strip()
        
        # For demonstration purposes, we'll return a dummy result
        # In production, replace this with the actual script call
        # Uncomment the line below and replace with the actual script path
        # result = subprocess.check_output(['python', 'path/to/fake_news_model.py', tweet_id], text=True).strip()
        
        # Dummy implementation (replace with actual script call)
        # For demo, we'll consider even-numbered IDs as fake news and odd-numbered as real
        result = '1' if int(tweet_id) % 2 == 0 else '0'
        return result
    
    except Exception as e:
        print(f"Error running fake news detection: {e}")
        return None

def index(request):
    context = {
        'result': None,
        'tweet_id': None
    }
    
    if request.method == 'POST':
        twitter_url = request.POST.get('twitter_url', '')
        tweet_id = extract_tweet_id(twitter_url)
        
        if tweet_id:
            result = run_fake_news_detection(tweet_id)
            context['result'] = result
            context['tweet_id'] = tweet_id
    
    return render(request, 'detector/index.html', context)
