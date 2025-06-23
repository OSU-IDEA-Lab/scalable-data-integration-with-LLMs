import argparse
import os.path

from DatabaseUtils.ConfigReader import get_config
from DatabaseUtils.PromptBuilder import Serializers, Samplers
from DatabaseUtils.enums import Dataset, Hint
from DatabaseUtils import prompt_templates
from cmd_utils import get_class_dict, str2bool

OPTION_DELIMITER = "-"
ARG_DELIMITER = "+"
IS_CHAR = "="

OUTPUT_DIR = get_config()["paths"]["output"]

class ArgHelper(object):

    def __init__(self):
        '''
            Define arguments
        '''
        self._parser = argparse.ArgumentParser()
        self._opt_to_abbr = {}
        self._abbr_to_opt = {}
        self.args = None

        # Can be "ground_truth" in which case the ground truth is used; otherwise, maybe provide a file path and maybe some other procedure?
        # After the full set of clusterings, the model produces a predicted mapping file
        # Evaluate over those mappings. should be stored as part of the config so I can determine how to evaluate as well.
        self._parser.add_argument("--clusters", type=str, required=True,
                            help="The clustering file path or setting {ground_truth}")
        self._parser.add_argument("--dataset", type=str, required=True, choices=[d.value for d in Dataset],
                            # [d.value for d in Dataset],
                            help="The dataset name")
        self._parser.add_argument("--source", type=str, required=True, help="The source database")
        self._parser.add_argument("--target", type=str, required=True, help="The target database")
        self._parser.add_argument("--seed", type=int, required=True, help="The random seed")
        self._opt_to_abbr["seed"] = "sd"

        #   Prompt args
        self._template_dict = get_class_dict(prompt_templates)
        self._parser.add_argument("--prompt", type=str, required=True, choices=self._template_dict.keys(),
                                  help="The prompt template")
        self._opt_to_abbr["prompt"] = "prmpt"

        #   Serializer args
        self._serializer_dict = get_class_dict(Serializers)
        self._parser.add_argument("--serializer", type=str, required=True, choices=self._serializer_dict.keys(),
                                  help="Specifies how the databases are serialized for the prompt")
        self._opt_to_abbr["serializer"] = "dbs"
        self._parser.add_argument("--hints", nargs='+', choices=[h.value for h in Hint],
                            help="Specifies which additional hints are included in the prompt")
        self._opt_to_abbr["hints"] = "hnts"
        self._parser.add_argument("--pp", type=str2bool,
                            help="Whether database serializations are pretty-printed or not (true or false)")
        self._opt_to_abbr["pp"] = "pp"
        self._parser.add_argument("--shuffle_relations", type=str2bool,
                                  help="Whether to randomly shuffle relation order depending on seed")
        self._opt_to_abbr["shuffle_relations"] = "shuf_rel"
        self._parser.add_argument("--shuffle_attributes", type=str2bool,
                                  help="Whether to randomly shuffle attribute order depending on seed")
        self._opt_to_abbr["shuffle_attributes"] = "shuf_att"

        #   Sampler args
        self._sampler_dict = get_class_dict(Samplers)
        self._parser.add_argument("--sampler", type=str, choices=self._sampler_dict.keys(),
                                  help=f"Specifies how the data values are sampled. Must be specified if including "
                                 f"{Hint.SAMPLE_DATA.value} hint")
        self._opt_to_abbr["sampler"] = "smplr"
        self._parser.add_argument("--sample_size", type=int, help=f"How many data points should be sampled")
        self._opt_to_abbr["sample_size"] = "n"
        self._parser.add_argument("--weighted", type=str2bool,
                            help=f"Use weighted sampling (true or false)")
        self._opt_to_abbr["weighted"] = "wghtd"

        for key, value in self._opt_to_abbr.items():
            self._abbr_to_opt[value] = key

    def get_option_names(self):
        return [act.dest for act in self._parser._actions if act.dest != "help"]

    def get_template(self):
        return self._template_dict[self.args.prompt]()

    def get_serializer(self):
        hints = []
        if self.args.hints is not None:
            hints = [Hint(h) for h in self.args.hints]
        return self._serializer_dict[self.args.serializer](hints_to_include=hints, pretty_print=self.args.pp)

    def get_sampler(self):
        if self.args.sampler is None:
            return None
        else:
            return self._sampler_dict[self.args.sampler](sample_size=self.args.sample_size, weighted=self.args.weighted)

    def _sort_args(self):
        # Sort any list args
        for opt in self.args.__dict__:
            if type(self.args.__dict__[opt]) is list:
                self.args.__dict__[opt] = sorted(self.args.__dict__[opt])

    def parse_args_from_list(self, args: list[str]):
        self.args = self._parser.parse_args(args)
        self._sort_args()

    def parse_args_from_cmd_line(self):
        self.args = self._parser.parse_args()
        self._sort_args()

    def parse_args_from_filepath(self, filepath: str) -> dict[str, str]:
        """
        Parses arguments from a given file path and returns a dictionary of arguments.

        The function takes a file path and extracts components from it to populate a dictionary
        of arguments, including dataset, source, target, clusters, and additional options
        parsed from the filename. It processes the given file path and filename to extract and
        map abbreviations to their respective options.

        Args:
            filepath: The full path to the file containing encoded arguments. The file path should be in the
            following format:
                <DATASET_NAME>/(<OPTIONAL_DIR>/){0,n}<SOURCE_DATABASE>-to-<TARGET_DATABASE>
                    /<CLUSTERING_METHOD>/(<OPTIONAL_DIR>/){0,n}<FILENAME>

        Returns:
            A dictionary where keys are argument names and values are corresponding argument
            values extracted from the file path and filename.
        """

        def _pop_until_contains(directory_list: list[str], substring_test_list):
            substring_test_list = [s.lower() for s in substring_test_list]
            while len(directory_list) > 0:
                this_dir = directory_list.pop(-1).lower()
                for substring in substring_test_list:
                    if substring in this_dir:
                        return this_dir
            return None

        filepath_parts = os.path.normpath(filepath).split(os.path.sep)

        filename = filepath_parts.pop(-1)
        if filename[-5:] == ".json":
            filename = filename[:-5]
        if filename[:7] == "OUTPUT-":
            filename = filename[7:]

        # Find clusters setting, ignore optional directories
        clusters = _pop_until_contains(filepath_parts, ["ground_truth", "gav", "multimap_"])
        assert clusters is not None, f"Could not find clusters setting in filepath: {filepath}"

        databases = filepath_parts.pop(-1)
        source, target = databases.split("-to-")

        # Find dataset name, ignore optional directories
        dataset = _pop_until_contains(filepath_parts, list(Dataset.__members__))
        assert dataset is not None, f"Could not find dataset name in filepath: {filepath}"

        arg_dict = {"dataset": dataset, "source": source, "target": target, "clusters": clusters}

        for abbr, arg in [abbr_arg.split(IS_CHAR) for abbr_arg in filename.split(OPTION_DELIMITER)]:
            opts = arg.split(ARG_DELIMITER)

            if len(opts) == 1:
                opts = opts[0]

            arg_dict[self._abbr_to_opt[abbr]] = opts

        # self.parse_args_from_dict(arg_dict)

        return arg_dict

    def get_filepath(self) -> str:

        # Get the subdirectories first
        sub_dirs = f"{OUTPUT_DIR}/{self.args.DATASET}/{self.args.source}-to-{self.args.target}/{self.args.clusters}"

        filename_parts = []

         opt_order = ["prompt", "serializer", "hints", "pp", "shuffle_relations", "shuffle_attributes", "sampler", "sample_size", "weighted", "seed"]
        for key in opt_order:
            val = self.args.__dict__[key]

            if val is None:
                continue

            if type(val) is list:
                val = ARG_DELIMITER.join(val)
            filename_parts.append(f"{self._opt_to_abbr[key]}={val}")
        full_path = f"{sub_dirs}/{'-'.join(filename_parts)}.json"

        return full_path

if __name__ == '__main__':
    this = ArgHelper()
    file_in = "amalgam/a1-to-a2/ground_truth/prmpt=SQL-dbs=JSON-pp=True-sd=1380313354.json"
    this.parse_args_from_filepath(file_in)
    print(this.get_filepath())
    print(this.get_filepath() == file_in)

    # for option in this._parser._positionals._actions:
    #     if "--help" in option.option_strings:
    #         continue
    #
    #     assert len(option.option_strings) == 1, \
    #         f"ArgHelper does not support option_strings with multiple representations (i.e., {option.option_strings})"
    #
    #     opt_string = option.option_strings[0]
#
#
# this._parser._positionals._actions[1:][0].option_strings
#
#
