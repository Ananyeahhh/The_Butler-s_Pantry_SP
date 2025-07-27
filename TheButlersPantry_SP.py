from pulp import *


# Problem Data

products = ['Cake', 'Bread', 'Pastry']
stores = ['Donnybrook', 'Sandymount', 'Monkstown', 'Online']
scenarios = ['Low', 'Medium', 'High']
prob = {'Low': 0.3, 'Medium': 0.5, 'High': 0.2}

demand = {
    'Cake': {'Donnybrook': {'Low': 10, 'Medium': 12, 'High': 14},
             'Sandymount': {'Low': 8, 'Medium': 10, 'High': 12},
             'Monkstown': {'Low': 7, 'Medium': 9, 'High': 11},
             'Online': {'Low': 5, 'Medium': 7, 'High': 9}},
    'Bread': {'Donnybrook': {'Low': 12, 'Medium': 15, 'High': 18},
              'Sandymount': {'Low': 10, 'Medium': 13, 'High': 16},
              'Monkstown': {'Low': 9, 'Medium': 12, 'High': 14},
              'Online': {'Low': 6, 'Medium': 8, 'High': 10}},
    'Pastry': {'Donnybrook': {'Low': 7, 'Medium': 10, 'High': 12},
               'Sandymount': {'Low': 6, 'Medium': 8, 'High': 10},
               'Monkstown': {'Low': 5, 'Medium': 7, 'High': 9},
               'Online': {'Low': 4, 'Medium': 5, 'High': 6}}
}

p = {'Cake': 15, 'Bread': 12, 'Pastry': 10}
c = {'Cake': 5, 'Bread': 4, 'Pastry': 3}
w = {'Cake': 2, 'Bread': 1.5, 'Pastry': 1}
emissions = {'Donnybrook': 1.0, 'Sandymount': 1.2, 'Monkstown': 1.5, 'Online': 1.8}
e_cost = 0.5

avg_demand = {
    i: {j: sum(prob[s] * demand[i][j][s] for s in scenarios) for j in stores}
    for i in products
}


# EV Model

EV = LpProblem("EV_Model", LpMaximize)
x_EV = LpVariable.dicts("x_EV", [(i, j) for i in products for j in stores], lowBound=0)
sales_EV = LpVariable.dicts("sales_EV", [(i, j) for i in products for j in stores], lowBound=0)
waste_EV = LpVariable.dicts("waste_EV", [(i, j) for i in products for j in stores], lowBound=0)

for i in products:
    for j in stores:
        EV += sales_EV[i, j] <= x_EV[i, j]
        EV += sales_EV[i, j] <= avg_demand[i][j]
        EV += waste_EV[i, j] >= x_EV[i, j] - sales_EV[i, j]

EV += lpSum(
    p[i] * sales_EV[i, j] - c[i] * x_EV[i, j] - w[i] * waste_EV[i, j] - e_cost * emissions[j] * x_EV[i, j]
    for i in products for j in stores
)

EV.solve()
EV_val = value(EV.objective)
x_EV_val = {k: x_EV[k].varValue for k in x_EV}


# EEV Calculation

EEV = 0
for s in scenarios:
    for i in products:
        for j in stores:
            d = demand[i][j][s]
            x_val = x_EV_val[i, j]
            sold = min(x_val, d)
            waste = max(0, x_val - d)
            EEV += prob[s] * (p[i] * sold - c[i] * x_val - w[i] * waste - e_cost * emissions[j] * x_val)


# Stochastic Model

SP = LpProblem("Stochastic_Model", LpMaximize)
x = LpVariable.dicts("x", [(i, j) for i in products for j in stores], lowBound=0)
sales = LpVariable.dicts("sales", [(i, j, s) for i in products for j in stores for s in scenarios], lowBound=0)
waste = LpVariable.dicts("waste", [(i, j, s) for i in products for j in stores for s in scenarios], lowBound=0)

for i in products:
    for j in stores:
        for s in scenarios:
            SP += sales[i, j, s] <= x[i, j]
            SP += sales[i, j, s] <= demand[i][j][s]
            SP += waste[i, j, s] >= x[i, j] - sales[i, j, s]

SP += lpSum(prob[s] * (
    p[i] * sales[i, j, s] - c[i] * x[i, j] - w[i] * waste[i, j, s] - e_cost * emissions[j] * x[i, j]
) for i in products for j in stores for s in scenarios)

SP.solve()
Stochastic_Optimal = value(SP.objective)


# Perfect Information Model

perfect_info_total = 0
for s in scenarios:
    model_s = LpProblem(f"Perfect_Info_{s}", LpMaximize)
    x_s = LpVariable.dicts("x_s", [(i, j) for i in products for j in stores], lowBound=0)
    sales_s = LpVariable.dicts("sales_s", [(i, j) for i in products for j in stores], lowBound=0)
    waste_s = LpVariable.dicts("waste_s", [(i, j) for i in products for j in stores], lowBound=0)

    for i in products:
        for j in stores:
            model_s += sales_s[i, j] <= x_s[i, j]
            model_s += sales_s[i, j] <= demand[i][j][s]
            model_s += waste_s[i, j] >= x_s[i, j] - sales_s[i, j]

    model_s += lpSum(
        p[i] * sales_s[i, j] - c[i] * x_s[i, j] - w[i] * waste_s[i, j] - e_cost * emissions[j] * x_s[i, j]
        for i in products for j in stores
    )

    model_s.solve()
    perfect_info_total += prob[s] * value(model_s.objective)


# Final Results

VSS = Stochastic_Optimal - EEV
EVPI = perfect_info_total - Stochastic_Optimal

print(f"EV (Expected Value): {EV_val:.2f}")
print(f"EEV (Expected from EV plan): {EEV:.2f}")
print(f"Stochastic Optimal Solution: {Stochastic_Optimal:.2f}")
print(f"Perfect Info Solution: {perfect_info_total:.2f}")
print(f"VSS (Value of Stochastic Solution): {VSS:.2f}")
print(f"EVPI (Expected Value of Perfect Info): {EVPI:.2f}")
