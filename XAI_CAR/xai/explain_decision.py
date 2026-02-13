def explain(action, cam_mean):
    if action == "LEFT":
        return f"Turn LEFT – activation biased left ({cam_mean:.2f})"
    elif action == "RIGHT":
        return f"Turn RIGHT – activation biased right ({cam_mean:.2f})"
    elif action == "STRAIGHT":
        return "Go STRAIGHT – lane centered"
    else:
        return "No reliable lane detected"