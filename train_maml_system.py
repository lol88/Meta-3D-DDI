from models.data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from models.few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
model = MAMLFewShotClassifier(args=args, device=device,
                              im_shape=(2, 3,
                                        args.image_height, args.image_width))

# maybe_unzip_dataset(args=args)
maml_system = ExperimentBuilder(model=model, data=MetaLearningSystemDataLoader, args=args, device=device)
maml_system.run_experiment()
