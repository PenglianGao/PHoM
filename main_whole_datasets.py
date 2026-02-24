#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:46:46
@Description: main.py
'''

from torch.distributed import algorithms
from utils.config import get_config
# from solver.unisolver import Solver
from solver.unisolver import Solver
# from solver.unisolver_finetune import Solver
from solver.testsolver import Testsolver
# from solver.gf_solver import Solver
# from solver.midnsolver import Solver
# from solver.innformersolver import Solver
import argparse,os,sys
#加载评估一体化的依赖
sys.path.append(os.path.join(os.path.dirname(__file__), 'py-tra'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from metrics_runner import run_metrics
from thop import profile
from costs_runner import run_costs


#训练测试一体化出结果_version
if __name__ == '__main__':
    # 外层循环，循环三次
    for i in range(2):


        parser = argparse.ArgumentParser(description='N_SR')
        parser.add_argument('--option_path', type=str, default='./option.yml')
        opt = parser.parse_args()
        # 配置文件列表，按顺序排列
        option_files = [
            'option_WV2.yml',
            'option_GF2.yml',
            'option_WV3.yml',]
        
        # algorithms = [
        # 'panmamba'
        # 'pannet',
        # 'srppnn',
        # 'innformer',
        #'sfinet',
        # 'panflownet',
        # 'hoif',
        # 'lformer'
        # ]
        # 遍历配置文件列表
        for option_file in option_files:
            # 获取每个配置文件的完整路径
            option_file_path = os.path.join(opt.option_path, option_file)
            print(f"Using config: {option_file_path}")
            
            cfg = get_config(option_file_path)
            # for algorithm in algorithms:
                # cfg['algorithm'] = algorithm
            # === 训练阶段 ===
            solver = Solver(cfg)
            log_name = solver.run()  # 修改 run() 返回 log_name

            # === 自动设置模型路径 ===    
            cfg['test']['algorithm'] = cfg['algorithm']
            cfg['test']['type'] = 'test'  
            cfg['test']['data_dir'] = cfg.get('data_dir_test') 
            cfg['test']['model'] = os.path.join(cfg['checkpoint'] + '/' + str(log_name), 'bestPSNR.pth')
            cfg['test']['save_dir'] = os.path.join(cfg['checkpoint'] + '/' + str(log_name),'result/')
            

            # === 测试阶段 ===
            test_solver = Testsolver(cfg)
            test_solver.run()

    
            data_eval = cfg.get('data_dir_test') 
            if data_eval.endswith('.h5'):
                path_ms = data_eval
                path_pan = data_eval
            else:
                path_ms = os.path.join(data_eval, 'ms')
                path_pan = os.path.join(data_eval, 'pan')
            path_predict = os.path.join(cfg['test']['save_dir'],'test')

            
            # 运行指标计算，如果失败会自动处理并继续执行
            metrics_success = run_metrics(path_ms, path_pan, path_predict, save_path=cfg['test']['save_dir'], cfg=cfg)

            if not metrics_success:
                print("⚠️  指标计算失败，但将继续执行后续流程...")
            run_costs(cfg)
