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
    Calls the AI model script for prediction.
    """
    try:
        result = subprocess.run([
            'python',
            os.path.join(os.path.dirname(__file__), 'gcn_predict.py'),
            str(tweet_id)
        ], capture_output=True, text=True)
        output = result.stdout.strip()
        if output in ['0', '1']:
            return output
        else:
            return None
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
