# =============================
# DISSERTATION CODE (MY WORK)
# Reddit data collection → preprocessing → hybrid sentiment → themes → geography → RQs → figures
# =============================

import praw
import pandas as pd
import datetime as dt
from tqdm import tqdm
import time
from prawcore.exceptions import Redirect, Forbidden, NotFound

# ===== REDDIT API CREDENTIALS =====
CLIENT_ID = "<CLIENT_ID>"
CLIENT_SECRET = "<CLIENT_SECRET>"
USER_AGENT = "Academic research for Brunel University"
USERNAME = "<USERNAME>"
PASSWORD = "<PASSWORD>"
# ================================

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD,
    user_agent=USER_AGENT
)

# =============================
# 1) DATA COLLECTION
# =============================

SUBREDDITS = [
    "climate", "environment", "science", "worldnews",
    "politics", "sustainability", "climatechange",
    "Futurology", "ClimateActionPlan", "GlobalWarming",
    "climateoffensive", "ecology", "Green"
]

KEYWORDS = [
    "climate", "global warming", "environment", "sustainability",
    "emissions", "carbon", "renewable", "clean energy",
    "climate policy", "SDG13", "climate action", "COP26",
    "climate finance", "adaptation", "mitigation"
]

START_DATE = dt.datetime(2019, 1, 1)
END_DATE = dt.datetime(2024, 7, 31)
TARGET_TOTAL = 50000

POSTS_PER_SUBREDDIT = 300
COMMENTS_PER_POST = 25
MIN_POST_SCORE = 10

comments = []
collected_count = 0

print(f"\n{'='*50}")
print(f"Collecting {TARGET_TOTAL} comments from {len(SUBREDDITS)} subreddits")
print(f"Keywords: {', '.join(KEYWORDS[:5])}...")
print("="*50)

for sub_name in SUBREDDITS:
    if collected_count >= TARGET_TOTAL:
        break

    try:
        print(f"\n[r/{sub_name}] Starting collection...")
        subreddit = reddit.subreddit(sub_name)

        posts = list(subreddit.top(time_filter="all", limit=POSTS_PER_SUBREDDIT))
        print(f"  Found {len(posts)} top posts")

        for submission in tqdm(posts, desc=f"Processing posts"):
            if collected_count >= TARGET_TOTAL:
                break

            if submission.score < MIN_POST_SCORE:
                continue

            post_date = dt.datetime.utcfromtimestamp(submission.created_utc)

            if START_DATE <= post_date <= END_DATE:
                try:
                    submission.comments.replace_more(limit=2)
                    comment_count = 0

                    for comment in submission.comments.list():
                        if collected_count >= TARGET_TOTAL or comment_count >= COMMENTS_PER_POST:
                            break

                        if comment.body.strip() and comment.author:
                            username = comment.author.name
                            anon_author = username[:2] + '***' + username[-2:] if len(username) > 4 else "Anonymous"

                            comments.append({
                                "subreddit": sub_name,
                                "author": anon_author,
                                "date": post_date.strftime('%Y-%m-%d'),
                                "text": comment.body,
                                "post_title": submission.title,
                                "post_score": submission.score
                            })
                            collected_count += 1
                            comment_count += 1

                    if collected_count % 1000 == 0:
                        print(f"  Collected: {collected_count}/{TARGET_TOTAL}")
                except Exception as e:
                    print(f"  Comment error: {str(e)[:50]}")
                time.sleep(0.3)

    except (Redirect, Forbidden, NotFound) as e:
        print(f"  Skipping r/{sub_name}: {str(e)}")
    except Exception as e:
        print(f"  Fatal error: {str(e)}")
        time.sleep(5)

if comments:
    df = pd.DataFrame(comments)
    if len(df) > TARGET_TOTAL:
        df = df.head(TARGET_TOTAL)

    filename = f"SDG13_50k_Comments_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)

    print(f"\n{'='*50}")
    print("COLLECTION COMPLETE!")
    print(f"Total comments: {len(df)}")
    print(f"Saved to: {filename}")
else:
    print("\nERROR: No comments collected!")

print("\nProcess completed.")


# =============================
# 2) PREPROCESSING
# =============================

import re
import spacy
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0
nlp = spacy.load("en_core_web_lg")

# Load the collected dataset (update this name to the actual file saved above)
df = pd.read_csv("SDG13_50k_Comments_YYYYMMDD_HHMM.csv", parse_dates=['date'])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

print("Cleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)

def detect_safe(text):
    try:
        if len(text) < 20:
            return 'unknown'
        return detect(text)
    except LangDetectException:
        return 'unknown'

print("Detecting language...")
from tqdm import tqdm
tqdm.pandas(desc="Language Detection")
df['language'] = df['cleaned_text'].progress_apply(detect_safe)
df = df[df['language'] == 'en'].copy()

def lemmatize_text(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha and len(token.lemma_) > 2]

print("Lemmatizing text...")
tqdm.pandas(desc="Lemmatization")
df['tokens'] = df['cleaned_text'].progress_apply(lemmatize_text)

df.to_csv("Preprocessed_Data.csv", index=False)
print(f"Preprocessing complete. Final dataset size: {len(df)} comments.")


# =============================
# 3) HYBRID SENTIMENT (VADER + CLIMATEBERT)
# =============================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch

df = pd.read_csv("Preprocessed_Data.csv")

vader = SentimentIntensityAnalyzer()
climate_bert = pipeline(
    "text-classification",
    model="climatebert/distilroberta-base-climate-f",
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
)

def hybrid_sentiment(text):
    vader_score = vader.polarity_scores(text)['compound']
    try:
        bert_result = climate_bert(text[:512])[0]
        pos_score = next((s['score'] for s in bert_result if s['label'] == 'POSITIVE'), 0)
        neg_score = next((s['score'] for s in bert_result if s['label'] == 'NEGATIVE'), 0)
        bert_score = pos_score - neg_score
    except Exception as e:
        print(f"ClimateBERT error on text: {e}")
        bert_score = 0.0

    hybrid_score = (vader_score * 0.4) + (bert_score * 0.6)
    return hybrid_score

print("Running hybrid sentiment analysis...")
tqdm.pandas(desc="Sentiment Analysis")
df['hybrid_sentiment'] = df['cleaned_text'].progress_apply(hybrid_sentiment)

df['sentiment_label'] = df['hybrid_sentiment'].apply(
    lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
)

df.to_csv("Data_With_Hybrid_Sentiment.csv", index=False)
print("Hybrid sentiment analysis complete and saved.")


# =============================
# 4) THEMATIC ANALYSIS (SDG 13 SUBTOPICS)
# =============================

import ast

df = pd.read_csv("Data_With_Hybrid_Sentiment.csv")

if isinstance(df['tokens'].iloc[0], str):
    df['tokens'] = df['tokens'].apply(ast.literal_eval)

THEMATIC_CATEGORIES = {
    'renewable_energy': ['solar', 'wind', 'renewable', 'energy', 'transition', 'clean', 'hydro', 'geothermal'],
    'climate_policy': ['policy', 'government', 'cop26', 'agreement', 'regulation', 'law', 'treaty', 'negotiation'],
    'climate_science': ['science', 'research', 'data', 'ipcc', 'study', 'evidence', 'scientist', 'model'],
    'individual_action': ['change', 'lifestyle', 'consumer', 'choice', 'personal', 'action', 'diet', 'travel'],
    'climate_justice': ['justice', 'equity', 'vulnerable', 'poor', 'fair', 'compensation', 'equality'],
    'technology_solutions': ['technology', 'innovation', 'solution', 'breakthrough', 'tech', 'ev', 'electric vehicle', 'battery'],
    'climate_impacts': ['disaster', 'flood', 'drought', 'fire', 'impact', 'effect', 'hurricane', 'extreme'],
    'economic_aspects': ['economy', 'cost', 'investment', 'finance', 'money', 'growth', 'tax', 'subsidy', 'price'],
    'international_agreements': ['paris', 'agreement', 'unfccc', 'cop', 'conference', 'parties', 'commitment']
}

def categorize_themes(token_list):
    found_themes = []
    for theme, keywords in THEMATIC_CATEGORIES.items():
        if any(keyword in token_list for keyword in keywords):
            found_themes.append(theme)
    return found_themes

print("Categorizing comments into themes...")
df['themes'] = df['tokens'].apply(categorize_themes)

df_exploded = df.explode('themes')
df_exploded = df_exploded.dropna(subset=['themes'])

df.to_csv("Data_With_Themes.csv", index=False)
df_exploded.to_csv("Data_With_Themes_Exploded.csv", index=False)
print("Thematic analysis complete. Saved exploded and unexploded versions.")


# =============================
# 5) GEOGRAPHY MENTIONS
# =============================

df = pd.read_csv("Data_With_Themes.csv")

geographical_entities = {
    'United States': ['usa', 'united states', 'u.s.a.', 'america', 'us', 'u.s.', 'new york', 'california', 'texas'],
    'Canada': ['canada', 'canadian', 'ontario', 'toronto', 'vancouver'],
    'European Union': ['eu', 'european union', 'europe', 'brussels', 'european'],
    'United Kingdom': ['uk', 'united kingdom', 'britain', 'england', 'london'],
    'Germany': ['germany', 'german', 'berlin', 'munich'],
    'France': ['france', 'french', 'paris'],
    'Australia': ['australia', 'australian', 'sydney', 'melbourne'],
    'China': ['china', 'chinese', 'beijing', 'shanghai'],
    'India': ['india', 'indian', 'delhi', 'mumbai'],
    'Africa': ['africa', 'african', 'kenya', 'nigeria', 'south africa'],
    'South America': ['south america', 'brazil', 'argentina', 'chile', 'amazon'],
    'Southeast Asia': ['southeast asia', 'indonesia', 'thailand', 'vietnam', 'philippines'],
    'Small Island Nations': ['small island', 'pacific island', 'caribbean', 'maldives', 'fiji'],
    'Global_South': ['developing country', 'global south', 'third world'],
    'Global_North': ['developed country', 'global north', 'first world']
}

all_geo_keywords = []
for region, keywords in geographical_entities.items():
    for keyword in keywords:
        all_geo_keywords.append((keyword, region))

def find_mentioned_regions(text):
    found_regions = set()
    if isinstance(text, str):
        text_lower = text.lower()
        for keyword, region in all_geo_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                found_regions.add(region)
    return list(found_regions)

print("Finding geographical mentions...")
df['mentioned_regions'] = df['cleaned_text'].apply(find_mentioned_regions)

df.to_csv("Data_With_Geography.csv", index=False)
print("Geographical analysis complete. Master dataset saved.")


# =============================
# 6) RESEARCH QUESTIONS (RQ1–RQ4)
# =============================

from scipy.stats import f_oneway, ttest_ind, pearsonr

df_themes = pd.read_csv("Data_With_Themes_Exploded.csv")
df_geo = pd.read_csv("Data_With_Geography.csv")

df_themes['tokens'] = df_themes['tokens'].apply(ast.literal_eval)
df_geo['mentioned_regions'] = df_geo['mentioned_regions'].apply(ast.literal_eval)

print("=== RESULTS (4 RESEARCH QUESTIONS) ===")

print("\n--- RQ1: Sentiment by Subtopics (ANOVA) ---")
subtopic_samples = []
for theme in df_themes['themes'].unique():
    sample = df_themes[df_themes['themes'] == theme]['hybrid_sentiment'].dropna()
    if len(sample) > 30:
        subtopic_samples.append(sample)
f_stat, p_value = f_oneway(*subtopic_samples)
print(f"ANOVA Result: F={f_stat:.3f}, p={p_value:.4f}")

print("\n--- RQ2: COP26 Impact Analysis ---")
def categorize_cop26_period(date):
    date = pd.to_datetime(date)
    cop26_start = pd.to_datetime('2021-11-01')
    cop26_end = pd.to_datetime('2021-11-12')

    if date < cop26_start - pd.Timedelta(days=30):
        return 'Pre-COP26 Baseline'
    elif date >= cop26_start and date <= cop26_end:
        return 'COP26 Conference'
    elif date > cop26_end and date <= cop26_end + pd.Timedelta(days=180):
        return 'Post-COP26 (6 months)'
    else:
        return 'Long-Term Post-COP26'

df_themes['cop26_period'] = df_themes['date'].apply(categorize_cop26_period)
cop26_sentiment = df_themes.groupby('cop26_period')['hybrid_sentiment'].agg(['mean', 'count', 'std']).round(3)
print(cop26_sentiment)

print("\n--- RQ3: Sentiment by Subreddit (ANOVA) ---")
subreddit_samples = []
for sub in df_themes['subreddit'].unique():
    sample = df_themes[df_themes['subreddit'] == sub]['hybrid_sentiment'].dropna()
    if len(sample) > 30:
        subreddit_samples.append(sample)
f_stat, p_value = f_oneway(*subreddit_samples)
print(f"ANOVA Result: F={f_stat:.3f}, p={p_value:.4f}")

print("\n--- RQ4: Global North vs Global South (t-test) ---")
def is_global_south(region_list):
    if not isinstance(region_list, list):
        return False
    south_indicators = ['India', 'China', 'Africa', 'South America', 'Southeast Asia', 'Small Island Nations', 'Global_South']
    return any(region in south_indicators for region in region_list)

def is_global_north(region_list):
    if not isinstance(region_list, list):
        return False
    north_indicators = ['United States', 'Canada', 'European Union', 'United Kingdom', 'Germany', 'France', 'Australia', 'Global_North']
    return any(region in north_indicators for region in region_list)

south_sentiment = df_geo[df_geo['mentioned_regions'].apply(is_global_south)]['hybrid_sentiment']
north_sentiment = df_geo[df_geo['mentioned_regions'].apply(is_global_north)]['hybrid_sentiment']

t_stat, p_value = ttest_ind(south_sentiment, north_sentiment, equal_var=False)
print(f"T-test Result: t={t_stat:.3f}, p={p_value:.4f}")


# =============================
# 7) VISUALIZATIONS (FIGURES)
# =============================

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sentiment_counts = df_themes['sentiment_label'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title('Distribution of Sentiment Labels in Climate Discourse')
plt.ylabel('Number of Comments')
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8))
community_sentiment = df_themes.groupby('subreddit')['hybrid_sentiment'].mean().sort_values()
community_counts = df_themes['subreddit'].value_counts()
valid_communities = community_counts[community_counts > 100].index
community_sentiment_filtered = community_sentiment[valid_communities]

plt.barh(range(len(community_sentiment_filtered)), community_sentiment_filtered.values)
plt.yticks(range(len(community_sentiment_filtered)), community_sentiment_filtered.index)
plt.title('Average Sentiment by Reddit Community (RQ3)')
plt.xlabel('Average Sentiment Score')
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('sentiment_by_community.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(14, 6))
cop26_means = []
cop26_counts = []

for period in ['Pre-COP26 Baseline', 'COP26 Conference', 'Post-COP26 (6 months)', 'Long-Term Post-COP26']:
    if period in df_themes['cop26_period'].values:
        period_data = df_themes[df_themes['cop26_period'] == period]['hybrid_sentiment']
        cop26_means.append(period_data.mean())
        cop26_counts.append(len(period_data))

cop26_data = {
    'Period': ['Pre-COP26 Baseline', 'COP26 Conference', 'Post-COP26 (6 months)', 'Long-Term Post-COP26'],
    'Mean_Sentiment': cop26_means,
    'Count': cop26_counts
}
cop26_df = pd.DataFrame(cop26_data)

x_pos = range(len(cop26_df['Period']))
plt.plot(x_pos, cop26_df['Mean_Sentiment'], marker='o', linestyle='-', linewidth=2, markersize=8)

plt.axvspan(1.1, 1.9, alpha=0.2, color='grey', label='COP26 Conference')
plt.xticks(x_pos, cop26_df['Period'], rotation=45, ha='right')
plt.ylabel('Mean Sentiment Score')
plt.title('Sentiment Trends Around COP26 Climate Conference (RQ2)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('cop26_timeline.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nDATA FILES CREATED:")
print("Raw: SDG13_50k_Comments_YYYYMMDD_HHMM.csv")
print("1) Preprocessed_Data.csv")
print("2) Data_With_Hybrid_Sentiment.csv")
print("3) Data_With_Themes.csv")
print("4) Data_With_Themes_Exploded.csv")
print("5) Data_With_Geography.csv")
