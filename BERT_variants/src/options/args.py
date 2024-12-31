import argparse

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Full model name", type=str, required=True)
    parser.add_argument("--config", help="Bert pretrained config", type=str, default="bert-large")
    parser.add_argument("--lora", help="lora finetune", action="store_true")
    parser.add_argument("--full_finetune", help="allow pretrained encoder trainable", action="store_true")
    
    parser.add_argument("--outdir", default='outdir')
    
    args = parser.parse_args()
    if args.config=="bert-base":
        from .config import ConfigBertBase as Config
    elif args.config=="bert-large":
        from .config import ConfigBertLarge as Config
    elif args.config=="bert-explain-original":
        from .config import ConfigBertLargeExplain_Original as Config
    elif args.config=="bert-explain-2Outputs":
        from .config import ConfigBertLargeExplain_2Outputs as Config
    elif args.config=="bert-explain-Hcls":
        from .config import ConfigBertLargeExplain_Hcls as Config
    else:
        assert (0), "Invalid config"
    
    return args, Config