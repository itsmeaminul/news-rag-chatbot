# 📖 User Usage Guide - News Article Chatbot

A simple guide to help you get started with the Bangladesh News RAG Chatbot and ask effective questions.

## 📑 Table of Contents

- [Getting Started](#getting-started)
- [Interface Overview](#interface-overview)
- [Setting Up Your First Session](#setting-up-your-first-session)
- [Query Types & Examples](#query-types-and-examples)
- [Quick Tips](#quick-tips)

---

## Getting Started

### What is this chatbot?

The Bangladesh News RAG Chatbot helps you explore and analyze news from major Bangladeshi outlets:
- **Prothom Alo** (English)
- **The Daily Star**
- **BDNews24**

Ask questions in natural language and get AI-powered responses based on the latest news articles.

### Before you start:
- ✅ Application is running in your browser
- ✅ You have internet connection
- ✅ Modern web browser (Chrome, Firefox, Safari, Edge)

---





## Interface Overview

### Main Layout

![Bangladesh News RAG Chatbot Interface](data\assets\app-screenshot.png)

*Screenshot: Main interface of the Bangladesh News RAG Chatbot*

### Sidebar Buttons

| Button | What it does | When to use it |
|--------|--------------|----------------|
| **🔄 Scrape Recent News** | Gets latest articles from news sites | First time using, or daily for fresh news |
| **📝 New Chat** | Start a new conversation | When changing topics completely |
| **📈 Database Stats** | Shows how many articles you have | Check if you have enough news data |
| **🗑️ Reset Database** | Deletes all articles (ask for confirmation) | If you want to start fresh |

---

## Setting Up Your First Session

### Step 1: Get News Articles

1. **Click "🔄 Scrape Recent News"**
2. **Wait for it to finish** (usually 2-5 minutes)
   
   You'll see progress messages like:
   ```
   Scraping news articles... 
   This may take a few minutes.
   ✅ Scraped 101 articles!
   Indexing articles...
   ```

3. **Check your data** - Look at "📈 Database Stats":
   ```
   📊 Database Statistics
   • Total Articles: 101
   • Unique Sources: 3
   • Sources:
      • daily_star
      • prothom_alo
      • bdnews24
   ```

### Step 2: Test It Out

Try asking: **"How many articles do you have?"**

If you get a good response with numbers, you're ready to go! 🎉

---

## Query Types and Examples

### 1. **Article Counts**
*Find out what data is available*

**Ask things like:**
```
"How many articles do you have?"
"How many sports news articles?"
"Count of Daily Star articles"
"How many articles about cricket?"
```

**You'll get responses like:**
```
I have 447 article chunks from 3 news sources in the database.
Available Sources: • The Daily Star • Prothom Alo • BDNews24
```


## Quick Tips

### ✅ Good Questions:
- Use simple, clear language
- Ask one main thing at a time
- Include relevant keywords
- Be specific about topics you want

### ❌ Avoid:
- Very long, complicated questions
- Multiple unrelated topics in one question
- Very technical jargon

### 🔄 If You Don't Get Good Results:
1. Try simpler words
2. Check if you have enough articles (📈 Database Stats)
3. Scrape recent news (🔄 Scrape Recent News)
4. Start a new chat (📝 New Chat)

---

**That's it! You're ready to explore recent news with AI assistance.** 🚀

*Need fresh news? Remember to click "🔄 Scrape Recent News" daily for the latest articles.*