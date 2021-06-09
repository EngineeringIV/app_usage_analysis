# app_usage_analysis
 Analyze app usage and categorize apps based on app and user information
 
 ## Analysis Motivation
 For this project, I was interested in grouping apps and users based on both app information (name and overview) and user usage information (daily usage pattern). Some questions that I had in mind are:
 1. What are some of the gaming apps in the app list
 2. What are some groups of users that share common behaviors
 3. Any other interesting findings

## File Structure
- Python files
  - main.py: main execution file
  - utils.py: helper functions
  - group_dashboard.py: dashboard to visualize results and facilitate grouping

- Data files (In "Data" folder)
  - app_information.csv: app information
  - app_information.pkl: pickle version of app_information.csv
  - user_app_usage.csv: usage information
  - user_app_usage.pkl: pickle version of user_app_usage.pkl
  - category_lut_new.csv: label/sublabel - topic lookup table
  - df_similarity.pkl: pickle file of app similarity matrix. Input to the dashboard
  - df_usage.pkl: pickle file of processed usage data. Input to the dashboard

- Txt file
  - requirements.txt: packages required to run the code
 
- Auxillary files
  - app_user_analysis.ipynb: a step-by-step explanation and demo of my thought process and design logic. (note: codes are presented differently for easier demonstration)
  - Design Thoughts.pdf: a summary document for high-level design logic, code design, instruction, and future improvements
 
 ## Techniques and Overall Design Logic
 The overall code design logic follows the logic as follow:

- Data pre-processing
- NLP topic modeling (for App)
  - Latent Dirichlet Allocation (LDA)
  - Non-negative Matrix Factorization (NMF)
- Similarity calculation (for App)
- Unsupervised clustering (App or user)
  - KMeans (with normalization and principle component analysis)
- Post process.

## Analysis Results
1. With NLP topic modeling, I selected topics whose top key words contain the word 'game' or gaming-related words ('shooter', 'role play') as game-related topics. On top of NLP topic modeling, I used a simplified version of collaborative filtering to determine top 10 similar apps for an app A and assigned the most common label among the 10 apps as the topic for app A (if it was unclassified from the NLP topic modeling). I then combined these two results together to an updated dataframe called df_nlp. Those apps in this dataframe with label 'Game' are game apps (or game-related such as steam, origin, roblox, or battlenet)

2. Since there are many possible combinations with the data we have (app label/sub-label + daily mins, app label/sublabel + app counts), I decided to write a plotly dash based dashboard to filter data as desired. In that dashboard, I can filter app label, sublabel and determine the parameter I'd like to look at (in terms of number of apps used or daily mins spent). I can also determine the groups I'd like to see

   - Passionate: whose app count or daily average time spent are more than 75 percentile
   - Above average: whose app count or daily average time spent are within 50 percentile to 75 percentile
   - Ordinary: whose app count or daily average time spent are within 25 percentile to 50 percentile
   - Sleeper: whose app count or daily average time spent are below 25 percentile

The dashboard also gives the ability to choose an app and see the top 10 similar/related apps and they yield some interesting results (see Question 3). The dashboard can be modified fairly easily to accomodate further analysis and it has significant flexibilities in determining groups of users. For demonstration purpose, I will use code to extract passionate shooting game (FPS or TPS) users who spent a lot of time on shooting games users who passionate about both gaming apps and video-related apps.

3. Since my code has the ability to extract the top 10 similar apps for every single app, I can make the arguement that the top 10 similar apps are the most related apps. For a game platform such as Origin or Steam, the top similar games should be the top games played on the platform. The same can go for non-game apps like Zoom, whose user are mostly students or people working from home and I'd expect to see some office or productivity apps as the most similar apps. The results below confirm out hypothesis:

   - Origin: Top games are The Sims, Battlefield V, and FIFA
   - Blizzard Battlenet: Top games are World of Warcraft, Overwatch, and Hearthstone
   - Steam: Top games are CSGO, League of Legends (wrongly classfied unfortunately), and GTA
   - Twtich (steaming platform): Top games streamed are World of Warcraft, Fortnite, and Minecraft
   - Zoom: Browser, Office (Powerpoint, Excel, Word), and Social (WhatsApp, Spotify, Slack)
