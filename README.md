## Title: Stock Market News-Pattern Intelligence Agent

## Overview

This project builds an AI agent that explains why a stock moved a certain way by connecting news sentiment to price patterns.
Given a stock ticker and date range, the agent:

1. Fetches relevant financial news

2. Extracts sentiment + important events using LLM structured analysis

3. Pulls market price data and detects common technical patterns

4. Correlates specific news events with bullish/bearish movements

6. Produces an easy-to-read report summarizing insights

It transforms raw news to trading intelligence.

## Reason for picking up this project

I have always had a liking and inquistivity for financial markets. This project seemed like the perfect opportunity to work on something like this. 

As for how all the course elements will be covered, prompting & structured output will be used for pydantic schemas for sentiment & event extraction, semantic search to filter and fetch relevant market news, a RAG will be used for financial news retrieval and glossary contexts. Tool calling will be used for web search and market data API. Most importantly LangGraph will be used for multi-node workflow orchestration and persistent state.

## Video Summary Link: 

Make a short -  3-5 min video of yourself, put it on youtube/googledrive, and put its link in your README.md.

- you can use this free tool for recording https://screenrec.com/
- Video format should be like this:
- your face should be visible
- State the overall job of your agent: what inputs it takes, and what output it gives.
- Very quickly, explain how your agent acts on the input and spits out the output. 
- show an example run of the agent in the video


## Plan

I plan to execute these steps to complete my project.

- [DONE] Step 1 involves setting up the GitHub repository, adding a basic project structure, and writing an initial README with the project overview and goals.
- [DONE] Step 2 involves defining the LangGraph state schema (ticker, date range, articles, sentiments, patterns, final report) using TypedDict models.
- [DONE] Step 3 involves implementing a news fetching step that uses a web/news API or tool to retrieve articles for a given stock ticker and date range, with basic error handling.
- [DONE] Step 4 involves designing prompt templates and structured output schemas for sentiment and event extraction from each news article.
- [DONE] Step 5 involves creating a sentiment extraction node that runs the LLM on each article and stores structured sentiment + event tags in the LangGraph state.
- [DONE] Step 6 involves implementing market data fetching (price history for the ticker over the same date range) and simple technical pattern detection logic (e.g., bullish/bearish moves, breakouts, reversals).
- [DONE] Step 7 involves writing correlation logic that maps news sentiment/events to detected market patterns (e.g., negative sentiment around dates where a bearish reversal occurs).
- [DONE] Step 8 involves building a report generation node that turns the correlated data into a readable, structured textual report for the user.
- [TODO] Step 9 involves connecting all nodes into a complete LangGraph workflow with proper state transitions from user input → news → analysis → patterns → report.
- [TODO] Step 10 involves running an end-to-end example with a real stock ticker and date range, verifying correctness, and adjusting prompts/logic as needed.
- [TODO] Step 11 involves adding robust error handling, logging, and handling edge cases (no news found, API failures, insufficient data) across the workflow.
- [TODO] Step 12 involves polishing documentation, updating the README with usage instructions and the video summary link, and writing the final conclusion reflecting on what was achieved.

## Conclusion:

After completion.



  