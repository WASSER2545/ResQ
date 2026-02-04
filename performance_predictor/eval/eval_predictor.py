from eval.eval_scalar_date import *
from eval.eval_scalar_str import *
from eval.eval_scalar_num import *

def eval_predict(res, agg_metrics):
    eval_card = []
    if not res['eval_type_cnt']:
        return 0
    for node in agg_metrics:
        if "EvalScalar" in node["name"]:
            output_rows = node["output_rows"]
            eval_card.append(output_rows)
    simple_eval = res['eval_type_cnt'][0]["simple"]
    simple_predict = num_predict_from_scalar(simple_eval, eval_card[0])
    
    date_eval = res['eval_type_cnt']["date"]
    date_predict = date_predict_from_scalar(date_eval, eval_card[0])
    
    str_eval = res['eval_type_cnt']["str"]
    str_predict = str_predict_from_scalar(str_eval, eval_card[0])
    
    return simple_predict + date_predict + str_predict