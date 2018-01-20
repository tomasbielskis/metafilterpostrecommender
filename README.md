Metafilter.com (MeFi) post recommender 

What's MeFi?

MetaFilter is a community weblog where users post interesting content found on the web and engage in discussions about it. The community prides itself in exposing a diverse set of posts about what's best and most interesting on the internet. MeFi has over 70,000 registered users and has been around for almost 20 years. Despite having a very engaged and active user base, the site is fairly basic and old-school, using no automated content recommendation systems beyond popularity.  

Problem statement

Like many online communities and sites, MeFi harbors a wealth of content far exceeding what is possible for an individual user to explore. Popular posts tend to get a lot more exposure than the less noticed but equaly deserving ones hiden in the long tail. This self-fulfilling quality of popularity is an evil that we ought to fight! 

Mission statement

With my project, I attempted to facilitate content discovery on MeFi and circumvent the trouble caused by post popularity by quantifying the tastes and preferences of individual users and aligning them with the explicit and implicit characteristics of posts and comments. I wanted to produce a recommender that is superior to collaborative filtering based methodoly based on favorites alone.

Process

1. Scrape all content from the site. 
2. Parse all posts and comments.
3. Combine with publicly available MeFi metadata. 
4. Natural language processing:
  a) Stemming all text
  b) Converting words into vectors based on term frequency (TF) and inverse document frequency (IDF)
  c) Non-negative matrix factorization (NMF) and latent dirichlet allocation (LDA) to compress the word features into a smaller number of latent features
5. User preference feature engineering:
  a) 4 sources of signal on user tastes: 
    - Posts written by the user
    - Posts favorited by the user
    - Comments written by the user
    - Comments favorited by the user
  b) Collect and combine text features for the four categories above.
6. Identify the posts that have features closest to the user preferences using a cosine similarity matrix 

Results

I had two main goals: 
  1. Quantity. Increase the number of users that can benefit from my methodology compared to collaborative filtering based on favorite ratings. 
  2. Quality. Improve the quality of the recommendations compared to the base case of picking random posts. 

My recommender is able to create a set of post recommendations for 32,000 MeFi users, which is about half of the user base and covers each active user, defined by having written or favorited at least one post or comment. This is 4 times better than the reach of a collaborative filtering recommender using the favorite ratings, since 2014. 

Also, my methodology has 100 better recall* score, and 5 times better precision* (using a sample of top 5 recommendations) compared to picking at random on a set of posts that I reserved to test the model and the actual past favorite data. 

*Precision is the number of posts in the recommendation set that were actually favorited by users, divided by the number of recommendations given based on the test set of posts. 

*Recall is the number of posts in the recommendation set that were actually favorited by users, divided by the number of total actual favorites in the test set of posts. 

Discussion

My original idea was to develop a collaborative filtering model. However, after exploring the data I realized that the favorites ratings are incredibly sparse. 

 
 
