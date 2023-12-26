import gurobipy as gp


def mtp(m):
    # Assuming you have a Gurobi model named 'model'
    # Initialize flags
    is_mip = False
    is_qcp = False

    # Check for integer or binary variables
    if any(var.vType != gp.GRB.CONTINUOUS for var in m.getVars()):
        is_mip = True

    # Check for quadratic constraints
    if m.getQConstrs():
        is_qcp = True

    # Determine the type of model
    if is_mip and is_qcp:
        print("This is a Mixed-Integer Quadratically Constrained Program (MIQCP)")
    elif is_mip:
        print("This is a Mixed-Integer Linear Program (MILP)")
    elif is_qcp:
        print("This is a Quadratically Constrained Program (QCP)")
    else:
        print("This is a Linear Program (LP)")
