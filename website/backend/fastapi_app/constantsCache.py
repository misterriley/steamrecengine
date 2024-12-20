def find_constant_value(constants_list, name):
    for d in constants_list:
        if d["name"] == name:
            return d["value"]

    raise ValueError("Constant named " + name + " not found")


class ConstantsCache:
    def __init__(self, constants_list):
        self.IMPUTE_SEQ_LENGTH = find_constant_value(constants_list, "IMPUTE_SEQ_LENGTH")

        self.STATUS_LESS = find_constant_value(constants_list, "STATUS_LESS")
        self.STATUS_MORE = find_constant_value(constants_list, "STATUS_MORE")
        self.STATUS_IGNORE = find_constant_value(constants_list, "STATUS_IGNORE")

        self.N_REGR_TERMS = find_constant_value(constants_list, "N_REGR_TERMS")
        self.REGR_TERM_PLSR = find_constant_value(constants_list, "REGR_TERM_PLSR")
        self.REGR_TERM_DIFF = find_constant_value(constants_list, "REGR_TERM_DIFF")
        self.REGR_TERM_LOG_LENGTH = find_constant_value(constants_list, "REGR_TERM_LOG_LENGTH")
        self.REGR_TERM_LOGIT = find_constant_value(constants_list, "REGR_TERM_LOGIT")
        self.REGR_TERM_YEAR = find_constant_value(constants_list, "REGR_TERM_YEAR")
        self.REGR_TERM_LOG_TOTAL_RATINGS = find_constant_value(constants_list, "REGR_TERM_LOG_TOTAL_RATINGS")

        self.COMP_STATUS_STARTED = find_constant_value(constants_list, "COMP_STATUS_STARTED")
        self.COMP_STATUS_FINDING_BEST_REGRESSIONS = find_constant_value(constants_list, "COMP_STATUS_FINDING_BEST_REGRESSIONS")
        self.COMP_STATUS_FINDING_BEST_CLASSIFIER = find_constant_value(constants_list, "COMP_STATUS_FINDING_BEST_CLASSIFIER")
        self.COMP_STATUS_SORTING_PREDICTIONS = find_constant_value(constants_list, "COMP_STATUS_SORTING_PREDICTIONS")
        self.COMP_STATUS_FINISHED = find_constant_value(constants_list, "COMP_STATUS_FINISHED")
        self.COMP_STATUS_ERROR = find_constant_value(constants_list, "COMP_STATUS_ERROR")

        self.ALPHA = find_constant_value(constants_list, "ALPHA")
        self.BETA = find_constant_value(constants_list, "BETA")
        self.NUM_SLOPE = find_constant_value(constants_list, "NUM_SLOPE")
        self.DENOM_SLOPE = find_constant_value(constants_list, "DENOM_SLOPE")

        self.TAG_TERM_BONUS_MULTIPLIER = find_constant_value(constants_list, "TAG_TERM_BONUS_MULTIPLIER")
        self.MAX_GAMES_TO_SORT = find_constant_value(constants_list, "MAX_GAMES_TO_SORT")



