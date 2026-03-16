import pyomo.environ as pyo


def build_model(data, risk_weight=0.0, alpha=0.1):
    """Build a two-stage stochastic LP for tomato processing.

    risk_weight : 0 = risk-neutral (maximise E[profit])
                  1 = fully risk-averse (maximise CVaR)
    alpha       : tail probability for CVaR (e.g. 0.1 = 90 % CVaR)
    """
    scenarios = data['scenarios']
    probs = data['probabilities']
    products = list(data['conversion'].keys())

    model = pyo.ConcreteModel()

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    model.SCEN = pyo.Set(initialize=scenarios)
    model.PROD = pyo.Set(initialize=products)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    model.prob = pyo.Param(model.SCEN, initialize=dict(zip(scenarios, probs)))
    model.available = pyo.Param(model.SCEN, initialize=data['available'])
    model.price = pyo.Param(
        model.SCEN, model.PROD,
        initialize={(s, p): data['prices'][s][p]
                    for s in scenarios for p in products}
    )
    model.conv = pyo.Param(model.PROD, initialize=data['conversion'])
    model.proc_cost = pyo.Param(model.PROD, initialize=data['proc_cost'])
    model.capacity = pyo.Param(model.PROD, initialize=data['capacity'])

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    # First-stage: capacity reservation (tons of final product)
    model.reserve = pyo.Var(model.PROD, domain=pyo.NonNegativeReals)

    # Second-stage: actual production per scenario
    model.produce = pyo.Var(model.SCEN, model.PROD,
                            domain=pyo.NonNegativeReals)

    # Sales (≤ production; all may not always be sold)
    model.sales = pyo.Var(model.SCEN, model.PROD, domain=pyo.NonNegativeReals)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def cap_reserve_rule(m, s, p):
        return m.produce[s, p] <= m.reserve[p]
    model.cap_reserve_con = pyo.Constraint(
        model.SCEN, model.PROD, rule=cap_reserve_rule)

    def capacity_rule(m, s, p):
        return m.produce[s, p] <= m.capacity[p]
    model.capacity_con = pyo.Constraint(
        model.SCEN, model.PROD, rule=capacity_rule)

    def tomato_balance_rule(m, s):
        total_used = sum(m.produce[s, p] * m.conv[p] for p in m.PROD)
        return total_used <= m.available[s]
    model.tomato_balance = pyo.Constraint(model.SCEN, rule=tomato_balance_rule)

    def sales_rule(m, s, p):
        return m.sales[s, p] <= m.produce[s, p]
    model.sales_limit = pyo.Constraint(model.SCEN, model.PROD, rule=sales_rule)

    # ------------------------------------------------------------------
    # Helper: scenario profit expression
    # ------------------------------------------------------------------
    def scenario_profit_expr(m, s):
        return sum(
            m.price[s, p] * m.sales[s, p] - m.proc_cost[p] * m.produce[s, p]
            for p in m.PROD
        )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    if risk_weight > 0:
        model.var_profit = pyo.Var(model.SCEN, domain=pyo.Reals)
        model.VaR = pyo.Var(domain=pyo.Reals)
        model.CVaR_slack = pyo.Var(model.SCEN, domain=pyo.NonNegativeReals)

        def profit_def_rule(m, s):
            return m.var_profit[s] == scenario_profit_expr(m, s)
        model.profit_def = pyo.Constraint(model.SCEN, rule=profit_def_rule)

        def cvar_rule(m, s):
            return m.var_profit[s] + m.CVaR_slack[s] >= m.VaR
        model.cvar_con = pyo.Constraint(model.SCEN, rule=cvar_rule)

        expected_slack = sum(
            model.prob[s] * model.CVaR_slack[s] for s in model.SCEN
        )
        cvar_expr = model.VaR - (1.0 / alpha) * expected_slack
        expected_profit_expr = sum(
            model.prob[s] * model.var_profit[s] for s in model.SCEN
        )
        model.obj = pyo.Objective(
            expr=(1 - risk_weight) * expected_profit_expr +
            risk_weight * cvar_expr,
            sense=pyo.maximize,
        )
    else:
        expected_profit_expr = sum(
            model.prob[s] * scenario_profit_expr(model, s) for s in model.SCEN
        )
        model.obj = pyo.Objective(
            expr=expected_profit_expr, sense=pyo.maximize)

    return model


def solve_model(model, solver='glpk'):
    slv = pyo.SolverFactory(solver)
    results = slv.solve(model, tee=False)
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        return model, results
    else:
        raise RuntimeError(
            f"Solver did not find an optimal solution. "
            f"Status: {results.solver.termination_condition}"
        )


def extract_results(model, data):
    """Pull all decision variables into plain Python dicts / DataFrames."""
    import pandas as pd

    scenarios = list(model.SCEN)
    products = list(model.PROD)
    probs = data['probabilities']

    reserves = {p: pyo.value(model.reserve[p]) for p in products}

    production = pd.DataFrame(index=scenarios, columns=products, dtype=float)
    sales_df = pd.DataFrame(index=scenarios, columns=products, dtype=float)
    for s in scenarios:
        for p in products:
            production.loc[s, p] = pyo.value(model.produce[s, p])
            sales_df.loc[s, p] = pyo.value(model.sales[s, p])

    # Compute scenario profits
    scenario_profits = {}
    for s in scenarios:
        profit = sum(
            pyo.value(model.price[s, p]) * pyo.value(model.sales[s, p])
            - pyo.value(model.proc_cost[p]) * pyo.value(model.produce[s, p])
            for p in products
        )
        scenario_profits[s] = profit

    expected_profit = sum(probs[i] * scenario_profits[s]
                          for i, s in enumerate(scenarios))

    return {
        'reserves': reserves,
        'production': production,
        'sales': sales_df,
        'scenario_profits': scenario_profits,
        'expected_profit': expected_profit,
    }
