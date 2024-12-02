def check_model_device_type(model):
    try:
        if hasattr(model, "device"):
            return model.device
        elif hasattr(model, "parameters"):
            return next(model.parameters()).device
            
    except Exception as e:
        print(f"Could not determine device for {model}: {e}")
        return None
    

def ignoreWarnings():
    import warnings
    import os
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    # Monkey-patch the warnings.warn function
    original_warn = warnings.warn
    warnings.warn = lambda *args, **kwargs: None
    return 
