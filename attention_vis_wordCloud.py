from AWDEncClas.predict_with_classifier import *
from attention_visualization import *
from AWDEncClas.eval_clas import attentionLexicBased


#lsName = ["imdb","yelp"]
#lsclas = ["POS", "NEG"]
#
#d={}
#i=1
#texts = []
####     [IMDB POS]  ###############
######################################
## [mistake] #
#text = """This sounded like a really interesting movie from the blurb. Nazis, occult , government conspiracies. I was expecting a low budget Nazi version of the DaVinci code or the Boys from Brazil or even Shockwaves. Instead you get something quite different, more psychological, more something like from David Lynch. That was actually a plus. But the way the story is told is just awful.<br /><br />Part of the trouble is the casting. Andrienne Barbeau's character starts off the moving being somewhat timid and afraid. She just doesn't do that well, even at her age, though she certainly tried. The actor cast as the son apparently thought this was a comedy. Most of the other actors also seemed to have thought this was a campy movie, or at least acted like it, rather than simply being quirky. The only one that I thought did really well was the daughter, Siri Baruc.<br /><br />Another big part is the pacing. It starts off very slowly. So slowly you might be tempted to turn it off. But then it gets compelling for a while when you get to the daughter's suicide and the aftermath. But shortly afterward, it all becomes a jumbled mess. Some of this was on purpose, but much of it was just needlessly confusing, monotonous, and poorly focused.<br /><br />The real problem, is it's simply not a pleasant movie to watch. It's slow, dull, none of the characters are likable. Overuse of imagery and sets. Some movies you see characters get tortured. In this, it's the viewer that does. It does have a few creepy moments, most notably the creepy Nazi paintings and the credits, but the rest of the movie is mostly just tiresome."""
#d[lsName[0]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#text = "This is a documentary unlike any other. It has so many layers and shows us so much that trying to analyze it all at once is nearly impossible. Documentarian William Greaves shows us the process of film-making from a different perspective. We see the struggles of the actors, the director, the sound crew, and everybody else trying to hang in there and make this film successful. If this was just about a movie being made it would be ordinary. What Greaves does is make it more complex by having a crew film the actors, and then this will be filmed by another crew, only to have another crew film the whole thing. Three cameras, each with a different goal. It has an almost dizzying affect on you but at the same time is exciting. I like the parts where the crew organizes together and discusses what is going on. Even they are somewhat in the dark as to what Greaves is trying to do. Half see this as an experiment while the other half sees it as a chaotic and confusing failure. No matter what side you choose, you can't argue that Greaves doesn't get you involved in this process."
#d[lsName[0]+str(i)+"_"+lsclas[0]] = text
#i += 1
#
#text = "Madonna gets into action, again and she fails again! Who's That Girl was released just one year after the huge flop of Shangai Surprise and two after the successful cult movie Desperately seeking Susan. She chose to act in it to forget the flop of the previous movie, not suspecting that this latter could be a flop, too. The movie received a bad acceptance by American critic and audience, while in Europe it was a success. Madonna states that \"Some people don't want that she's successful both as a pop star and a movie-star\". The soundtrack album, in which she sings four tracks sells well and the title-track single was agreat hit all over the world, as like as the World Tour. The truth isthat Madonna failed as an actress 'cause the script was quite weak. Butit's not so bad, especially for those who like the 80's: it's such a ramshackle, trash, colorful and joyful action movie ! At the end, it's very funny to watch it."
#d[lsName[0]+str(i)+"_"+lsclas[0]] = text
#i += 1
#
#
#text = "Though not a Greek I have had a lifelong interest in the Eastern Empire. Its fall in 1453 was the Greatest loss to Christianity in its entire history. Yet while the Easter Empire is not a topic much discussed in American intellectual circles, the US did not merely mimic Golden Byzantiums public architecture, the US is much absorbed in the fated Byzantine historical cycle and now has faced many of the crises involving certain people of a middle eastern extraction about whom it is said that there is a slight tendency for excessive exuberance on religious matters which humbled Great Byzantium. I wonder if the loss of the ability to speak plainly was the first sign post on the road to disaster.<br /><br />John Romer is to be credited not only for his excellent production but also for his joyful enthusiasm for the subject which is most refreshing.<br /><br />Not recommended for Americans who like political correctness."
#d[lsName[0]+str(i)+"_"+lsclas[0]] = text
#i += 1
#
#
####     [IMDB Neg]   ################
######################################
#text = "The movie contains a very short scene of Deneuve in a bathtub. She looks absolutely stunning for a lady age 56, but this is the only saving grace of the movie. Otherwise, it has a mindless, unmotivated script and the lead actress has none of Deneuve's appeal. The director apparently watched too many Peter Greenaway films and Pola X comes across as a student's imitation of the Greenaway style, without any of his inspiration."
#d[lsName[0]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#
#text = "\"Love and Human Remains\" is one of those obviously scripted, obviously acted, obviously staged flicks which is so obvious that the escape velocity from its contrivances and fabrication is beyond me. Not worth explaining, this amateurish flick tries to cram every clever line, every misanthropic overtone, every peculiar sexual predilection into one film with an absence of concern for making the pieces fit. In short, sensationalistic crap without the sensation...which pretty much just leaves crap."
#d[lsName[0]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#
#text = "Van Sant copies Hitchcock's masterpiece shot for shot including some modern facets: a walkman, and nudity from Anne Heche. Unless you have a strong desire to see Ms. Heche naked there is absolutely NO reason to see this film instead of the original. Hitchcock's masterpiece is much better and Van Sant fails to realize that in hiding the nudity and the gore, the original shower scene is all the more terrifying. Ask Janet Leigh about that one. The acting is also much flatter and the technical aspects much less impressive."
#d[lsName[0]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#
## [NOT mistake] 4/10 #
#text = "\"The Next Karate Kid\" is a thoroughly predictable movie, just like its predecessors. Its predictability often results in a feeling of impatience on the viewer's part, who often wishes the story could move a little faster. Despite its lulls and its extreme familiarity, however, this fourth entry in the series is painless, almost exclusively because of the presence of Morita. He doesn't seem tired of his role, and he does inject some life and humor into the film, becoming the best reason for you to see it."
#d[lsName[0]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#
##cls, att_sc, outStr = predict_textIn('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl', 'data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT_lgsfx_condensedX_clas_1.h5', text)
#
#i=1
####     [Yelp POS]   ################
######################################
#
#text = "When there's this much hype about a place, it's hard to live up to, but they did a pretty damn good job. Maybe not the best sandwich I've ever had in my life, but if it was any random sandwich shop I'd happened to stumble into, it would definitely be five stars without a second thought.\n\nThe chili cheese fries were among the best I've ever had. I don't know what they do to them - it's not the chili that's particularly great, it's not the cheese that's particularly great, and it's not the fries that are particularly great. But somehow, putting them all together just turns it into unbeatable awesome.\n\nThe sandwich itself (I got the corned beef) was very good despite the fact that I hate coleslaw. But they don't mean the typical mayo-and-carrot-and-purple-onion coleslaw; they make it with vinegar and oil, so it's basically just like seasoned lettuce. They put a bit much of it on, which overwhelmed the rest of the ingredients at first, but just pull about half of the slaw off and you have a kick-ass sandwich. The hot sauce goes really well on it too. And for a pretty reasonable price, I might add. Normally, I'd expect it to be about twice as much with that kind of reputation and downtown location."
#d[lsName[1]+str(i)+"_"+lsclas[0]] = text
#i += 1
#
#
#text = "I wavered between 4 and 5 stars here. If the rating was strictly based on food, I'd go with 5, easily. But, it's based on service, too, which is why I had to knock one off. Here's the deal...\n\nI love Morton's. Its old-school interior and no-gimmicks shtick is awesome. Plus, of course, the food is fantastic. However, I wanted to branch out, and Ruth's Chris had some great reviews, so I utilized Yelp's oh-so-convenient Open Table reservation feature, and scored a table for two on a Saturday night. \n\nOur reservation was for 8:30. We arrived right on the dot, but were told to go to the bar and wait. So, my boyfriend and I both got cocktails -- he had Glenlivet, which is obvs hard to mess up, and I got the Peachy Pear-tini, which was delish. And then, we waited. And waited. Around 9, our table was ready. \n\nOur waitress was Kristen, and she was fantastic. Very attentive, friendly, witty, and knowledgeable about the food. Here's our food rundown:\n\n- Barbequed shrimp app: I initially thought was BBQ sauce + shrimp, but it ain't. It's succulent li'l shrimp in a deliciously garlicky, white wine-infused sauce with a cayenne kick. Kristen suggested that we dip the bread in the sauce after the shrimp were gone, and we happily took her up on that. \n\n-Ruth's chopped salad: Holy crap. The most memorable salad of my life. I rarely rave about salads. In fact, I think this is the first time I ever have in my life. But, this salad was incredible! It has crispy onion strings on top, hearts of palm (which I love), and this amazing lemon-basil dressing, among other delicious things. So awesome.\n\n- Filet mignon: My boyfriend and I both opted for the filet. I got mine with extra butter, he got his with a bleu cheese crust. Mine was phenomenal. The sizzling butter really adds an extra dimension of flavor that puts it beyond Morton's. My boyfriend found the bleu cheese to be a li'l too generous, and he wound up scraping some off, but still enjoyed it.\n\n- Broccoli au gratin: Again, phenomenal. I've never enjoyed broccoli so much. Granted, it was swimming in bubbling, gooey cheese...\n\n- Potatoes lyonnaise: We took a risk here, as many people rave about the garlic mashed potatoes, and we saw mountains of them carried past our table. But, it paid off big-time. These were, hands-down, the BEST lyonnaise potatoes EVER. Pamela's has nothing on these. Every mouthful was a medley of crusty brown potatoes and caramelized onion goodness. I dream about these potatoes.\n\nSo, based on that experience, we went back again the next Saturday. Different waitress, same food (literally, we ordered exactly the same items, 'cause they were so damn good). I don't remember the waitress' name that we had the second time, but we waited longer for food to appear, were checked on far less, she took the cocktail shaker with my extra martini in it before I'd gotten to pour it in my glass, our steaks were not done correctly (he got my medium-rare, I got his medium-well), and it took forever to get the check. I really think it ruins the experience when you're done eating and ready to go, and you're just sitting there, drumming your fingers, looking around the dining room and wondering, \"Where the hell is the waitress with our check?\" \n\nThat said, we will definitely be back. We'll just request to be seated in Kristen's section!"
#d[lsName[1]+str(i)+"_"+lsclas[0]] = text
#i += 1
#
#
#text = "I'm not the clubbing type of girl, but this club definitely blew me away. Thanks to my girlfriend, I was able to bypass the atrocious line that had already formed by 9 PM and be the first to step into the nightclub. Some of my other friends arrived at 8 PM and were first in line. I felt really bad for the boys in my group, though. They knew I hadn't gone clubbing in Vegas prior to this, so they forked over $50 cover each to accompany me and the girls (although I hear cover is only $30 during off season).\n\nI can't believe how beautifully decorated it is. The pool, cabanas, daybeds, and gogo dancers are absolutely amazing! Drinks are overpriced, of course. 5 shots of patron for $60. The venue didn't seem THAT huge, but my group of 20 somehow got separated an hour after we entered the club, and we weren't able to find each other for the rest of the night. The club was PACKED (I hear this is where everybody wants to be on a Saturday night).\n\nLamar Odom and Khloe Kardashian were there the same night, along with some other NBA players I didn't recognize. Definitely a good first Vegas clubbing experience =)"
#d[lsName[1]+str(i)+"_"+lsclas[0]] = text
#i += 1
#
#
#text = "If the \"price range\" still shows a bunch of dollar signs, it's wrong - this place is very reasonably priced!\n\nI've had nothing but good experiences coming here. The staff is friendly, the service is prompt, and they stand by their American values: Discounts and a special desk with discounts for men and women in uniform.\n\nMy dress shirts came back cleaned and pressed just right and pants were dry cleaned well too. You might not see this place from the street, but the service and value definitely stand out to me!"
#d[lsName[1]+str(i)+"_"+lsclas[0]] = text
#i += 1
#
#
####     [Yelp Neg]   ################
######################################
#
#text = "My husband had been bugging me to try this place for a while now, so I finally gave in. I am not a big waffle eater but I decided to get one and a coffee. I came in with my husband and daughter at 3pm on Good Friday. One person was working, a young girl. She asked what we wanted. I asked her to give me a minute, I had never been here before. So I got a waffle with bacon and maple icing, my husband got the cinammon roll waffle and my daughter got a plain waffle with whipped cream. I saw a waffle maker, but she put our pre-made waffles, not made to order, in a toaster oven type thing and heated it up. Maybe the waffle maker is just for show. I also ordered a coffee, no coffee in any container. I advised the girl and she said she would get some. So she proceeded to heat up our waffles, bring them to the table, make the two ladies behind us some coffee. At this point I was slightly frustrated. Not only did my waffle, which by the way is the size of a pop tart, taste like card board,it wasn't even hot.I could barely cut it with the plastic knife and fork. I went to the counter to tell the girl I no longer wanted my coffee, and I would like my money back. She states, It's almost ready. I also advised her the waffles were horrible. I asked her if I could get the manager's name and number. She said, he isn't here. I just gave up at this point. She did provide me with the owners name and the email for the business. I just think that for $4.50 a waffle, they would be made to order. I never did get my money back either. So we just walked out, without my coffee. We will definitely NOT be going back. They should also have planned better for a Holiday and get people in there that actually want to be there, and be helpful."
#d[lsName[1]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#
#text = "I love the idea of this place but I bought a groupon and you have to sign in on line within 30 days or it won't let you and they never answer the phone or return phone calls or email and when you go by no one is there I don't know how they keep running specials I suggest don't by a group on and the instructors aren't very pleasant to be around good luck I had to contact groupon to get my money back to purchase another if this happens to you group on is wonderful they will do what it is you want they will even contact tough lotus if you want ."
#d[lsName[1]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#
#text = "This place USED to be so great, what happened?? The last 3 times we went there we were very unimpressed and have pretty much given up. Each time the service was very slow, the help acts like they are overwhelmed and don't know what to do. But worse of all, the food has become ordinary! Something has definitely changed there and not for the better."
#d[lsName[1]+str(i)+"_"+lsclas[1]] = text
#i += 1
#
#
#text = "It seems like every other time I take my pets to Point Breeze there is an unnecessary frustration.   From being berating for using their online prescription program to being misquoted prices substantially to poor customer service I just keep hoping each time will be better.  Although I know they love animals I suspect they love money more.  Dr. Caroline. Simard is an excellent vet, but my last two visits have been pushed to other vets whose bedside manner is atrocious.   I'm a 40 year-old  who has been taking my pets to vets since childhood and I don't understand why each visit leaves a sour taste in my mouth."
#d[lsName[1]+str(i)+"_"+lsclas[1]] = text
#i += 1


#for key, value in d.items():
#    if "imdb" in key:
#        cls, att_sc, outStr = predict_textIn('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl',
#                                             'data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT_lgsfx_condensedX_clas_1.h5',
#                                             value)
#    else:
#        cls, att_sc, outStr = predict_textIn('data/nlp_clas/yelp_bi/yelp_bi_clas/tmp/itos.pkl',
#                                             'data/nlp_clas/yelp_bi/yelp_bi_clas/models/fwd_yelp_FT_ATT_lgsfx_clas_1.h5',
#                                             value)
#
#    att_lex = attentionLexicBased(outStr)
#    fileName = 'Figs/' + key + '.png'
#    fileName_lex = 'Figs/' + key + '_lex.png'
#
#    CreatWordCloud(outStr, list(att_sc[:, 0, 0]), fileName)
#    CreatWordCloud(outStr, att_lex, fileName_lex)
#
###############################################################################################
#################### [ large scale word cloud exp ] ###########################################
###############################################################################################
import pandas as pd
import random
import io

def getExamples(dataF, dataSetName):
    dRtrn = {}
    #i=1
    for r in dataF.itertuples():
        lab = lsclas[r[1]]
        dRtrn[dataSetName+str(r[0])+"_"+lab] = r[2]
        #i += 1
    return dRtrn



noExmp = 20
lsName = ["imdb","yelp"]
lsclas = ["NEG", "POS"]
d={}


imdb_df = pd.read_csv("data/nlp_clas/imdb/imdb_clas/test.csv", header=None, engine='python')
randIdx = random.sample(range(0,len(imdb_df)), noExmp)
d.update(getExamples(imdb_df.iloc[randIdx], lsName[0]))


yelp_df = pd.read_csv("data/nlp_clas/yelp_bi/yelp_bi_clas/test.csv", header=None, engine='python')
randIdx = random.sample(range(0,len(yelp_df)), noExmp)
d.update(getExamples(yelp_df.iloc[randIdx], lsName[1]))



for key, value in d.items():
    if "imdb" in key:
        cls, att_sc, outStr = predict_textIn('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl',
                                             'data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT_lgsfx_condensedX_clas_1.h5',
                                             value)
    else:
        cls, att_sc, outStr = predict_textIn('data/nlp_clas/yelp_bi/yelp_bi_clas/tmp/itos.pkl',
                                             'data/nlp_clas/yelp_bi/yelp_bi_clas/models/fwd_yelp_FT_ATT_lgsfx_clas_1.h5',
                                             value)

    # skip wrong decisions
    if cls and "NEG" in key:
        continue

    if not cls and "POS" in key:
        continue

    att_lex = attentionLexicBased(outStr)
    fileName = 'Figs/RealQuiz/' + key + '.png'
    fileName_lex = 'Figs/RealQuiz/' + key + '_lex.png'
    fileName_Mix = 'Figs/RealQuiz/' + key + '_Mix.png'

    logFileName = "Figs/RealQuiz/log.txt"

    with io.open(logFileName,"a", encoding="utf8") as f:
        f.write(key + "\t")
        if cls:
            f.write("POS\t")
        else:
            f.write("NEG\t")
        f.write(value + "\n")


    att_scr = np.exp(list(att_sc[:, 0, 0]))

    att_lex = np.exp(att_lex)/sum(np.exp(att_lex))

    att_mx = att_scr + att_lex
    att_mx = att_mx/sum(att_mx)

    CreatWordCloud(outStr, np.log(att_scr), fileName)
    CreatWordCloud(outStr, np.log(att_lex), fileName_lex)
    CreatWordCloud(outStr, np.log(att_mx), fileName_Mix)


