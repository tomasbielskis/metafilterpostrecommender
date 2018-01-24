# Metafilter.com Post Recommender

## What's MeFi?

MetaFilter (MeFi) is a community weblog where users post interesting content found on the web and engage in discussions about it. The community prides itself in exposing a diverse set of posts about what's best and most interesting on the internet. MeFi has over 70,000 registered users and has been around for almost 20 years. Despite having a very engaged and active user base, the site is fairly basic and old-school, using no automated content recommendation systems beyond popularity.  

## Problem Statement

Like many online communities and sites, MeFi harbors a wealth of content far exceeding what is possible for an individual user to explore. Popular posts tend to get a lot more exposure than the less noticed but equally deserving ones hiding in the long tail. This self-fulfilling quality of popularity is an evil that we ought to fight!

## Mission Statement

With my project, I attempted to facilitate content discovery on MeFi and circumvent the tunnel vision of popularity by quantifying the tastes and preferences of individual users and aligning them with the explicit and implicit characteristics of posts and comments. I wanted to produce a recommender that is superior to collaborative filtering methodology based on favorites alone.

## Process

1. Scrape all content from the site.
2. Parse all posts and comments.
3. Combine with publicly available MeFi metadata.
4. Natural language processing:
  a) stemming all text,
  b) converting words into vectors based on term frequency (TF) and inverse document frequency (IDF),
  c) non-negative matrix factorization (NMF) and latent dirichlet allocation (LDA) to compress the word features into a smaller number of latent features.
5. User preference feature engineering:
  - 4 sources of signal on user tastes derived from the available data:
    - Posts written by the user - 170k records
    - Posts favorited by the user - 1 million records
    - Comments written by the user - 7 million records
    - Comments favorited by the user - 15 million records
  - Collect and combine text features for the four categories above.
6. Identify the posts that have features closest to the user preferences by using a cosine similarity matrix.

## More Details in Plainer Lingo

Stemming refers to a common language processing technique that clips the word endings. For example, it turns the words: "joke," "joker," "joked", "joking" into "jok." This is a convenient shortcut to make the machine think all these words have similar meaning by turning them into the same token.

Term frequency and inverse document frequency refers to a process of turning words and documents into numbers and vectors based on how frequently the words occur and how many documents contain a particular word.

My vectorization also included groups of two words that frequently occur together, know as bigrams, as opposed to single words tokens, which are called unigrams. For example, New York, Bin Laden, music video.

NMF and LDA are methodologies that enable one to compress information content into a smaller set of abstract features. For example, I limited my vocabulary (most common unigrams and bigrams) to 50,000 word tokens, and then used NMF and LDA to reduce that vocabulary down to 50 categories of words that commonly occur together. I ended up with categories that expressed theme of the topic such as technology, politics, music, art, or emotional tone such as excited, angry, sad, and other similar abstractions.

Here are a few of these abstract features with the dominant words (the category labels are implicit, I arbitrarily added them for illustrative purposes only):

Game and video content:
game, play, video, music, youtub, releas, version, song, video game, super, sound, classic, games, featur, seri, theme, includ, music video, youtube, soundtrack

Movies:
film, short, movi, trailer, titl, documentari, short film, director, films, seri, featur, filmmak, star, produc, direct, hollywood, theater, mysteri, minut, stori

Common, casual talk:
peopl, know, want, like, don, think, just, someth, way, look, sure, realli, need, someon, feel, ll, thing, probabl, talk, wonder

Astronomy:
space, happen, slvimeo, station, report, earth, plan, flight, planet, nasa, launch, plane, moon, shuttl, pilot, orbit, space shuttl, bomb, near, protect

Comics:
comic, tell, chris, blog, stori, superhero, strip, writer, interview, work, comic book, cartoon, random, come, san, unexpect, superman, jame, spider, featur

Design:
just, use, make, creat, effect, design, model, matter, laugh, build, techniqu, water, market, engin, power, sold, devic, replac, process, human

NSFW / irreverent:
slyt, slyt nsfw, nsfw, vs, tom, robot, perform, kiss, song, guy, skeleton, sung, yeah, funni, sing, danc, walk, bird, william, girl

Inspiration:
time, long, travel, hour, puzzl, truli, word, befor, turn, got, long time, start, know, need, ny, hours, tri, use, inventor, home

Loss:
pass, talk, away, yes, win, 2011, steve, live, pass away, previous, job, steve job, years, appl, star, home, los, contest, offici, park

MeFi related:
previous, mefi, view, link, point, projects, project, mefi projects, interact, author, person, map, metafilter, futur, previous mefi, origin, thank, interview, run, known

## Tech Stack

- Python libraries: numpy, pandas, sklearn, nltk, multiprocessing, flask
- MongoDB, AWS EC2, S3, Elastic Beanstalk

## Results

I had two main goals:
  1. Quantity. Increase the number of users that can benefit from my methodology compared to collaborative filtering based on favorite ratings.
  2. Quality. Improve the quality of the recommendations compared to the base case of picking random posts.

My recommender is able to create a set of post recommendations for 32,000 MeFi users, which is about half of the user base and covers each active user, defined by having written or favorited at least one post or comment. This is 4 times better than the reach of a pure collaborative filtering recommender using the favorite ratings, since 2014.

Also, my methodology has 100 better recall* score, and 5 times better precision* (using a sample of top 5 recommendations) compared to picking at random on the set of posts that I reserved to test the model and the actual past favorite data.

* Precision is the number of posts in the recommendation set that were actually favorited by users, divided by the number of recommendations given based on the test set of posts.

* Recall is the number of posts in the recommendation set that were actually favorited by users, divided by the number of total actual favorites in the test set of posts.

## Discussion

My original idea was to develop a collaborative filtering model and complement it with text based post features. However, after exploring the data I realized that the favorites ratings for posts are incredibly sparse... I had less than 1,000,000 favorites since 2014, for 170,000 posts and 70,000 users, which divides up to a sparsity of less than a basis point (<0.01%). This basically killed the collaborative filtering idea and took me back to the drawing board.

The epiphany came when I realized that I can use text features for posts and comments that the user interacted with (either producing or favoriting them) to indicate their preferences. Most importantly this allowed me to expand and enrich my data set with features not only from posts but from comments as well, which were much more plentiful!  

## Future Plans

For my recommender, I only used the main part of MeFi, and none of the subsites. But the methodology should translate very easily to the rest of the site. In particular, this would be extremely effective and interesting applied to the Ask part of MeFi. If there is interest in the community I might build that in the future.

Given the time constraints for the project (only two weeks) and the number of items to process (7 million comments), I only ran three text processing models: NMF using Frobenius Norm, NMF using Kullback-Leibler divergence and LDA, my feature set was maxed at 50k and I extracted 50 latent features for each item, using unigrams and bigrams. I only used Snowball stemmer and no lemmatization. Hence, there is a lot of room left to optimize these parameters and explore the more interesting natural language processing techniques such as part of speech tagging and sentiment analysis.

Even though I tested my model on a synthetic set of randomly selected past posts and compared the performance metrics to random selection, the ultimate test for products like this is in the wild. It would be very interesting to calibrate the recommender based on user feedback in an AB testing or reinforcement learning framework.

The favoriting feature of MeFi is binary and not sufficiently popular to be a great indicator of user interest and the level of it. The main problem is that the absence of favorite does not necessarily indicate that the user would not like the post; in most cases, it just means that the user hasn't read it. What would be particularly useful in understanding user preferences is clickthrough data for posts and embedded links but I'm not sure if MeFi even tracks that at all.

Link to the recommender:
[www.metafilterpostrecommender.com](http://www.metafilterpostrecommender.com)
