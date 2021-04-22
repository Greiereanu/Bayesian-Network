#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import numpy as np

from pomegranate import *

numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')


# In[19]:


Smoker = DiscreteDistribution({ 'True': 0.25 , 'False': 0.75})
Age = DiscreteDistribution({ 'Less35': 0.47 , 'More35': 0.53})
Gender = DiscreteDistribution({ 'F': 0.49 , 'M': 0.51})


# In[34]:


Hypertension = ConditionalProbabilityTable(
    [[ 'Less35', 'F', 'True', 'True', 0.31 ],
     [ 'Less35', 'F', 'False', 'True', 0.24 ],
     [ 'Less35', 'M', 'True', 'True', 0.32 ],
     [ 'Less35', 'M', 'False', 'True', 0.27 ],
     [ 'More35', 'F', 'True', 'True', 0.39 ],
     [ 'More35', 'F', 'False', 'True', 0.34 ],
     [ 'More35', 'M', 'True', 'True', 0.33 ],
     [ 'More35', 'M', 'False', 'True', 0.3 ],
     [ 'Less35', 'F', 'True', 'False', 0.69 ],
     [ 'Less35', 'F', 'False', 'False', 0.76 ],
     [ 'Less35', 'M', 'True', 'False', 0.68 ],
     [ 'Less35', 'M', 'False', 'False', 0.73 ],
     [ 'More35', 'F', 'True', 'False', 0.61 ],
     [ 'More35', 'F', 'False', 'False', 0.66 ],
     [ 'More35', 'M', 'True', 'False', 0.67 ],
     [ 'More35', 'M', 'False', 'False',0.7 ]], [Age, Gender, Smoker] )

Diabetes = ConditionalProbabilityTable([['Less35', 'F', 'True', 0.21 ],
                                        ['Less35', 'M', 'True', 0.27 ],
                                        ['More35', 'F', 'True', 0.39 ],
                                        ['More35', 'M', 'True', 0.45 ],
                                        [ 'Less35', 'F', 'False', 0.79],
                                        ['Less35', 'M', 'False', 0.73 ],
                                        ['More35', 'F', 'False', 0.61 ],
                                        ['More35', 'M', 'False', 0.55 ]],
                                        [Age, Gender])
Anaemia = ConditionalProbabilityTable([['F', 'True', 'True', 0.38 ],
                                        ['F', 'False', 'True', 0.16 ],
                                        ['M', 'True', 'True', 0.21 ],
                                        ['M', 'False', 'True', 0.11 ],
                                        [ 'F', 'True', 'False', 0.62],
                                        ['F', 'False', 'False', 0.84 ],
                                        ['M', 'True', 'False', 0.79 ],
                                        ['M', 'False', 'False', 0.89 ]],
                                        [Gender, Diabetes])

Death = ConditionalProbabilityTable(
    [[ 'True', 'True', 'True', 'True', 0.97 ],
     [ 'True', 'True', 'False', 'True', 0.65 ],
     [ 'True', 'False', 'True', 'True', 0.6 ],
     [ 'True', 'False', 'False', 'True', 0.3],
     [ 'False', 'True', 'True', 'True', 0.55 ],
     [ 'False', 'True', 'False', 'True', 0.3 ],
     [ 'False', 'False', 'True', 'True', 0.25 ],
     [ 'False', 'False', 'False', 'True', 0.3 ],
     [ 'True', 'True', 'True', 'False', 0.03 ],
     [ 'True', 'True', 'False', 'False', 0.35 ],
     [ 'True', 'False', 'True', 'False', 0.4 ],
     [ 'True', 'False', 'False', 'False', 0.7 ],
     [ 'False', 'True', 'True', 'False', 0.45],
     [ 'False', 'True', 'False', 'False', 0.7],
     [ 'False', 'False', 'True', 'False', 0.75 ],
     [ 'False', 'False', 'False', 'False', 0.97 ]], [Anaemia, Diabetes, Hypertension] )


# In[35]:


s0 = State( Smoker, name="Smoker" )
s1 = State( Age, name="Age" )
s2 = State( Gender, name="Gender" )
s3 = State( Hypertension, name="Hypertension" )
s4 = State( Diabetes, name="Diabetes")
s5 = State(Anaemia, name="Anaemia")
s6 = State(Death, name="Death")


# In[36]:


network = BayesianNetwork( "Health" )
network.add_nodes(s0, s1, s2, s3, s4, s5, s6)

network.add_edge(s0, s3)
network.add_edge(s1, s3)
network.add_edge(s2, s3)

network.add_edge(s1, s4)
network.add_edge(s2, s4)

network.add_edge(s2, s5)
network.add_edge(s4, s5)

network.add_edge(s5, s6)
network.add_edge(s4, s6)
network.add_edge(s3, s6)


# In[37]:


network.bake()


# In[38]:


print (network.probability(np.array(['True', 'More35', 'M', 'True', 'True', 'True', 'True' ] , ndmin =2)))
#Probabilitatea de a deceda, stiind ca: esti fumator, > 35 ani, esti barbat, ai toate cele 3 boli
#0.0020441065837499997


# In[39]:


print (network.probability(np.array(['False', 'More35', 'M', 'False', 'False', 'False', 'False' ] , ndmin =2)))
#Probabilitatea de a nu deceda, stiind ca: nu esti fumator, > 35 ani, esti barbat, nu ai nicio boala.
#0.0673798096125


# In[51]:


observations = { 'Gender' : 'F'}
beliefs = map( str, network.predict_proba( observations ) )
print("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))
#folosit pentru a vedea probabilitatea de a nu avea diabet, fiind femeie


# In[52]:


observations = { 'Diabetes' : 'False'}
beliefs = map( str, network.predict_proba( observations ) )
print("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))
#Folosit pentru a vedea probabilitatea de a fi anemic, stiind ca nu esti diabetic
