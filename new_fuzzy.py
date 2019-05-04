import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("fuzzy_input.csv")

dataset_X = dataset.iloc[:, 0].values
dataset_Y = dataset.iloc[:, 1].values
dataset_Z = dataset.iloc[:, 2].values

print(dataset_X)
print(dataset_Y)
print(dataset_Z)



x = ctrl.Antecedent(np.arange(0,1,0.1), 'x')
y = ctrl.Antecedent(np.arange(0,1,0.1), 'y')
z = ctrl.Antecedent(np.arange(0,1,0.1), 'z')

epilepsy = ctrl.Consequent(np.arange(0, 1, 0.1), 'epilepsy')
x.automf(3)
y.automf(3)
z.automf(3)

x.view()
y.view()
z.view()
plt.show()

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
epilepsy['low'] = fuzz.trimf(epilepsy.universe, [0, 0.25, 0.5])
epilepsy['medium'] = fuzz.trimf(epilepsy.universe, [0.25, 0.5, 0.75])
epilepsy['high'] = fuzz.trimf(epilepsy.universe, [0.5, 0.75, 1])

# You can see how these look with .view()

epilepsy.view()
plt.show()


rule1 = ctrl.Rule(x['poor'] & y['average'] & z['good'], epilepsy['low'])
rule2 = ctrl.Rule(x['average'] & y['average'] & z['average'], epilepsy['medium'])
rule3 = ctrl.Rule(x['poor'] & y['poor'] & z['good'], epilepsy['high'])
#rule1.view()

epilepsyping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

epilepsyping = ctrl.ControlSystemSimulation(epilepsyping_ctrl)



# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)


#out_1 = np.where(out > 0,1,-1)
#out_1 = out_1.T
#np.savetxt("foo_4.csv", out_1, delimiter=",")



#for i in range(0,dataset_x.size):
epilepsyping.input['x'] = dataset_X[4]
epilepsyping.input['y'] = dataset_Y[4]
epilepsyping.input['z'] = dataset_Z[4]
# Crunch the numbers
epilepsyping.compute()

dataset_check = pd.read_csv("foo_4.csv")

dataset_true = dataset_check.iloc[:, 0].values
dataset_pred = dataset_check.iloc[:, 1].values
from sklearn.metrics import accuracy_score,confusion_matrix
#print(confusion_matrix(y_test,predictions))

print(confusion_matrix(dataset_true,dataset_pred))
print(accuracy_score(dataset_true,dataset_pred)*100)



print(epilepsyping.output['epilepsy'])
epilepsy.view(sim=epilepsyping)
plt.show()

