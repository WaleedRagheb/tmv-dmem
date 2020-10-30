import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from AWDEncClas.significant_test import *

#acc_att = eval_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT__NoCont2',attention=True)
#acc_withoutAtt = eval_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT')

acc_att = eval_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 3, lm_id='yelp_FT', clas_id='yelp_FT_ATT_lgsfx',attention=True)
acc_withoutAtt = eval_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 3, lm_id='yelp_FT', clas_id='yelp_FT')

print(acc_att)
print(acc_withoutAtt)

from scipy import stats

f_value, p_value = stats.f_oneway(acc_att,acc_withoutAtt)

print('F1: ' + str(f_value)+ ' p-value: ' + str(p_value))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')

data_to_plot = [ acc_withoutAtt, acc_att ]
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

# Save the figure
fig.savefig('fig1.png', bbox_inches='tight')

print('boxplot saved....')
