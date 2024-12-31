from .customedBert import CustomedBert
from .Self_Explain_Bert_H_cls import ExplainableModel_Hcls
from .Self_Explain_Bert_2outputs import ExplainableModel_2Outputs
from .Self_Explain_Bert_Ori import ExplainableModel_Original


def get_model_from_name(name, kwargs):
    if name=='bert':
        return CustomedBert(**kwargs)
    elif name == 'self_explain_original':
        return ExplainableModel_Original(**kwargs)
    elif name=='self_explain_2Outputs':
        return ExplainableModel_2Outputs(**kwargs)
    elif name=='self_explain_Hcls':
        return ExplainableModel_Hcls(**kwargs)
    else:
        assert(0), "Model not found"