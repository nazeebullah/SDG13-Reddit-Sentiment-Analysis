# 🌍 Quantifying Public Sentiment Toward Climate Action (SDG 13)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange.svg)
![SDG13](https://img.shields.io/badge/UN%20SDG-13%20Climate%20Action-green.svg)

## 📌 Project Overview
This repository contains the dataset and findings from my MSc Data Science and Analytics dissertation at Brunel University London (2024-2025). 

Understanding how the public genuinely feels about climate change targets is notoriously difficult using traditional surveys. This project bridges the gap in understanding genuine public engagement by conducting a large-scale computational analysis of public sentiment toward climate action through the lens of Reddit discourse.

## 🔬 Methodology: Hybrid NLP Framework
To accurately capture the nuances of internet discourse regarding climate change, this study utilizes a novel hybrid sentiment analysis framework:
* **VADER:** Optimized for social media context, capturing slang, emojis, and sentiment intensity.
* **ClimateBERT:** A domain-specific transformer model fine-tuned for climate-related text and scientific terminology.

The study analyzed **46,630 English-language Reddit comments** scraped from 10 climate-relevant communities between January 2019 and July 2024. The pipeline includes preprocessing, thematic categorization (e.g., individual action, economic aspects), and geographic entity extraction.

## 📂 Dataset Structure
https://onedrive.live.com/?cid=2c55e5acb935c8a8&id=2C55E5ACB935C8A8!s381245bd33b84c9a95b97e28b4e1c539&resid=2C55E5ACB935C8A8!s381245bd33b84c9a95b97e28b4e1c539&ithint=folder&e=HN3lxe&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy8yYzU1ZTVhY2I5MzVjOGE4L0lnQzlSUkk0dURPYVRKVzVmaWkwNGNVNUFjajh3cFZWQ1JrYW1WYjZhTm81bnhBP2U9SE4zbHhl
The included data files represent various stages of the NLP pipeline:
* `SDG13_50k_Comments_20250808_0154.csv`: The raw dataset scraped from Reddit.
* `Preprocessed_Data.csv`: Cleaned text data with extracted NLP tokens.
* `Data_With_Hybrid_Sentiment.csv`: Contains the calculated hybrid sentiment scores and labels.
* `Data_With_Themes.csv` & `Data_With_Themes_Exploded.csv`: Comments categorized by specific climate sub-themes.
* `Data_With_Geography.csv`: Extractions of specific regional and geographic mentions.

## 📄 Full Academic Report
The complete academic dissertation, detailing the methodology, literature review, and comprehensive findings, can be found in the attached PDF: [`Quantifying_Public_Sentiment_SDG13.pdf`](./Quantifying_Public_Sentiment_SDG13.pdf).

## 🎓 Academic Context
* **Degree:** MSc Data Science and Analytics
* **Institution:** Brunel University London
* **Academic Year:** 2024-2025
