#Credits to Lin Zhouhan(@hantek) for the complete visualization code
import random, os, numpy, scipy
from codecs import open

from clyent import color

from fastai.text import *
from sklearn.preprocessing import normalize

from wordcloud import WordCloud
import matplotlib.pyplot as plt




def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def processAttentionScores_norm(attScrs):
    att_sc = attScrs[3:]
    cutThrld = 1.0/ len(att_sc)

    #att_sc = [(-1 * np.log(sc)) if sc > cutThrld else 0.0 for sc in att_sc]

    #att_sc1 = [sc if sc > cutThrld else 0.0 for sc in att_sc]
    #att_sc_norm = att_sc1 / np.linalg.norm(att_sc1, ord=1)

    denm = max(att_sc)#-min(att_sc)

    att_sc2 = [(sc/denm) for sc in att_sc]
    #return softmax(att_sc1)

    #att_sc_norm = att_sc1 / np.linalg.norm(att_sc1, ord=1)
    #return att_sc2
    return att_sc


def processAttentionScores_inv (attScrs):
    att_sc_ = attScrs[3:]
    mx = max(att_sc_)
    att_sc = [mx-s for s in att_sc_]
    #tt_sc_norm = att_sc / np.linalg.norm(att_sc, ord=1)
    cutThrld = 1.0/ len(att_sc)

    #att_sc = [(-1 * np.log(sc)) if sc > cutThrld else 0.0 for sc in att_sc]

    #att_sc1 = [sc if sc > cutThrld else 0.0 for sc in tt_sc_norm]

    denm = max(att_sc)-min(att_sc)

    att_sc2 = [(sc/denm) for sc in att_sc]
    #return softmax(att_sc1)

    #att_sc_norm = att_sc1 / np.linalg.norm(att_sc1, ord=1)
    return att_sc2





def createHTML(texts, weights, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    #fileName = "visualization/"+fileName
    fOut = open(fileName, "w", encoding="utf-8")

    weights = processAttentionScores_norm(weights)

    tok = Tokenizer().proc_all_mp(partition_by_cores([texts]))

    print('Number of tok: ' + str(len(tok[0])) + ' - Number of scores: ' + str(len(weights)))
    print(fileName)
    tok_proc = [w.replace('\n', '\\n').replace('\'','\\\'').replace('\"','\\\"') for w in tok[0]]
    texts = " ".join(tok_proc)

    #weights = [x * 100 for x in weights]


    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
	
    for (var k=0; k < any_text.length; k++) {
		var tokens = any_text[k].split("~~");
		intensity = trigram_weights[k];
		var heat_text = "<p><br><b>Example:</b><br>";
		space = " ";
		for (var i = 0; i < tokens.length; i++) {
			
			heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
			
			if (space == "") {
			space = " ";
			}
		}
    //heat_text += "<p>";
		document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "%s"%x
    textsString = "var any_text = [\"%s\"];\n"%("".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [[%s]];\n"%(",".join(map(str, weights)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()
  
    return




###############################################################################################################
##############################################################################################################
##############################################################################################################


def processTextAll(textList):
    rtrn = []
    i=0
    print("ProcessText: ")
    for txt in textList:
        print(i)
        tok = Tokenizer().proc_all_mp(partition_by_cores([txt]))
        tok_proc = [w.replace('\n', '\\n').replace('\'', '\\\'').replace('\"', '\\\"') for w in tok[0]]
        texts = " ".join(tok_proc)
        rtrn.append(texts)
        i=i+1

    return rtrn

def processAttentionAll(attList):
    rtrn = []
    i=0
    print("processAttention: ")
    for att in attList:
        print(i)
        attProc = processAttentionScores_norm(att)
        rtrn.append(attProc)
        i=i+1

    return rtrn

def createHTMLALL(dataF, colorStr, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    # fileName = "visualization/"+fileName

    textAll = dataF.iloc[range(0, len(dataF)), 3]
    att_All = dataF.iloc[range(0, len(dataF)), 0]

    textAll = processTextAll(textAll)
    att_All = processAttentionAll(att_All)

    fOut = open(fileName, "w", encoding="utf-8")

   #weights = processAttentionScores_norm(weights)

   #tok = Tokenizer().proc_all_mp(partition_by_cores([texts]))

   #print('Number of tok: ' + str(len(tok[0])) + ' - Number of scores: ' + str(len(weights)))
   #print(fileName)
   #tok_proc = [w.replace('\n', '\\n').replace('\'', '\\\'').replace('\"', '\\\"') for w in tok[0]]
   #texts = " ".join(tok_proc)

    # weights = [x * 100 for x in weights]

    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = " var color = \""+colorStr+"\";" + """
    var ngram_length = 3;
    var half_ngram = 1;

    for (var k=0; k < any_text.length; k++) {
		var tokens = any_text[k].split(" ");
		intensity = trigram_weights[k];
		var heat_text = "<p><br><b>Example:</b><br>";
		space = " ";
		for (var i = 0; i < tokens.length; i++) {

			heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";

			if (space == "") {
			space = " ";
			}
		}
    //heat_text += "<p>";
		document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\"" % x
    textsString = "var any_text = [%s];\n" % (",".join(map(putQuote, textAll)))
    weightsString = "var trigram_weights = [%s];\n" % (",".join(map(str, att_All)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()

    return

###############################################################################################################
##############################################################################################################
##############################################################################################################
def processAll(textList, attentionList):
    rtrnTXT = []
    rtrnATT = []
    i=0
    print("ProcessText: ")
    for txt in textList:
        print(i)
        tok = txt.split("~~")
        tokSize = len(tok)
        tok_proc = [w.replace('\n', '\\n').replace('\'', '\\\'').replace('\"', '\\\"') for w in tok]

        texts = "~~".join(tok_proc[3:])
        rtrnTXT.append(texts)

        att = (attentionList[i])[0:tokSize]
        if type(att) is torch.Tensor():
            rtrnATT.append(torch.stack(att).cpu().numpy())
        else:
            rtrnATT.append(att)

        i=i+1

    return rtrnTXT, rtrnATT

def processAttentionAll_2(attList):
    rtrn = []
    i=0
    print("processAttention: ")
    for att in attList:
        print(i)
        attProc = processAttentionScores_norm(np.exp(att))
        rtrn.append(list(attProc))
        i=i+1

    return rtrn

def createHTMLALL_att(dataF, colorStr, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    # fileName = "visualization/"+fileName

    textAll = dataF.iloc[range(0, len(dataF)), 0].as_matrix()
    att_All = dataF.iloc[range(0, len(dataF)), 3].as_matrix()

    textAll, att_All = processAll(textAll, att_All)
    att_All = processAttentionAll_2(att_All)

    fOut = open(fileName, "w", encoding="utf-8")

   #weights = processAttentionScores_norm(weights)

   #tok = Tokenizer().proc_all_mp(partition_by_cores([texts]))

   #print('Number of tok: ' + str(len(tok[0])) + ' - Number of scores: ' + str(len(weights)))
   #print(fileName)
   #tok_proc = [w.replace('\n', '\\n').replace('\'', '\\\'').replace('\"', '\\\"') for w in tok[0]]
   #texts = " ".join(tok_proc)

    # weights = [x * 100 for x in weights]

    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = " var color = \""+colorStr+"\";" + """
    var ngram_length = 3;
    var half_ngram = 1;

    for (var k=0; k < any_text.length; k++) {
		var tokens = any_text[k].split("~~");
		intensity = trigram_weights[k];
		var heat_text = "<p><br><b>Example:</b><br>";
		space = " ";
		for (var i = 0; i < tokens.length; i++) {

			heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";

			if (space == "") {
			space = " ";
			}
		}
    //heat_text += "<p>";
		document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\"" % x
    textsString = "var any_text = [%s];\n" % (",".join(map(putQuote, textAll)))
    weightsString = "var trigram_weights = [%s];\n" % (",".join(map(str, att_All)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()

    return

def createHTMLALL_OneText(textAll,att_All, fileName, colorStr="0,255,0"):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    # fileName = "visualization/"+fileName

    #textAll = dataF.iloc[range(0, len(dataF)), 0].as_matrix()
    #att_All = dataF.iloc[range(0, len(dataF)), 3].as_matrix()

    textAll, att_All = processAll(textAll, att_All)
    #att_All = processAttentionAll_2(att_All)
    att_All[0] = list(np.exp(att_All[0][3:]))

    fOut = open(fileName, "w", encoding="utf-8")

   #weights = processAttentionScores_norm(weights)

   #tok = Tokenizer().proc_all_mp(partition_by_cores([texts]))

   #print('Number of tok: ' + str(len(tok[0])) + ' - Number of scores: ' + str(len(weights)))
   #print(fileName)
   #tok_proc = [w.replace('\n', '\\n').replace('\'', '\\\'').replace('\"', '\\\"') for w in tok[0]]
   #texts = " ".join(tok_proc)

    # weights = [x * 100 for x in weights]

    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = " var color = \""+colorStr+"\";" + """
    var ngram_length = 3;
    var half_ngram = 1;

    for (var k=0; k < any_text.length; k++) {
		var tokens = any_text[k].split("~~");
		intensity = trigram_weights[k];
		var heat_text = "<p><br><b>Example:</b><br>";
		space = " ";
		for (var i = 0; i < tokens.length; i++) {

			heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";

			if (space == "") {
			space = " ";
			}
		}
    //heat_text += "<p>";
		document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\"" % x
    textsString = "var any_text = [%s];\n" % (",".join(map(putQuote, textAll)))
    weightsString = "var trigram_weights = [%s];\n" % (",".join(map(str, att_All)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()

    return



##############################################################################

def CreatWordCloud(outStr, att_sc, filename):
    ex = ["t_up","i", "we", "he", "she", "of", "a", "as", "the", "this",
          "been", "be", "is", "are","was","were", "has", "have", "had", "do", "did","does",
          "n\'t", "\'s", "in", "and", "out",
          "br", "<br", "/><br", "/>"]

    tok = outStr.split("~~")
    tok_M = tok[3:]
    attsc_M = att_sc[3:]

    d = {}
    for idx,tk in enumerate(tok_M):
        if tk in ex or len(tk) < 2:
            continue
        if tk in d:
            d[tk] = np.max([d[tk], np.exp(attsc_M[idx])])
        else:
            d[tk] = np.exp(attsc_M[idx])

    wordcloud = WordCloud(width=900, height=500, relative_scaling=0.5, max_words=100,background_color="white").generate_from_frequencies(d)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filename)




