# # deployment.py
# import datetime
#
#
# class ModelDeployer:
#     """模型部署器"""
#
#     def __init__(self, model_instance):
#         self.model = model_instance
#         self.deployment_dir = None
#
#     def prepare_deployment(self, model_checkpoint=None):
#         """准备部署"""
#         if model_checkpoint is None:
#             if hasattr(self.model, 'best_checkpoint'):
#                 model_checkpoint = self.model.best_checkpoint
#             else:
#                 raise ValueError("未指定模型检查点")
#
#         # 检查是否已有SavedModel
#         savedmodel_path = os.path.join(model_checkpoint, 'saved_model')
#         if os.path.exists(savedmodel_path):
#             self.deployment_dir = savedmodel_path
#             print(f"✅ 使用已有的SavedModel: {savedmodel_path}")
#             return savedmodel_path
#
#         # 如果没有，重新导出
#         print("正在导出为SavedModel格式...")
#         export_path = os.path.join(model_checkpoint, 'saved_model_export')
#         os.makedirs(export_path, exist_ok=True)
#
#         # 重构模型并导出
#         if self.model._prediction_model is None:
#             self.model._prediction_model = self.model.reconstruct_model()
#
#         with self._suppress_output():
#             self.model._prediction_model.export(export_path)
#
#         self.deployment_dir = export_path
#         print(f"✅ SavedModel已导出到: {export_path}")
#         return export_path
#
#     def test_deployment(self, test_data):
#         """测试部署模型"""
#         if self.deployment_dir is None:
#             raise ValueError("请先调用 prepare_deployment()")
#
#         # 加载SavedModel
#         model = tf.saved_model.load(self.deployment_dir)
#         serve_fn = model.signatures['serve']
#
#         # 测试推理
#         result = serve_fn(*test_data)
#         return result
#
#     def create_deployment_package(self, output_dir='deployment_package'):
#         """创建完整的部署包"""
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 1. 复制模型
#         import shutil
#         model_dest = os.path.join(output_dir, 'model')
#         shutil.copytree(self.deployment_dir, model_dest)
#
#         # 2. 创建配置文件
#         config = {
#             'model_info': self.get_model_info(),
#             'deployment_date': datetime.datetime.now().isoformat(),
#             'api_endpoints': self._generate_api_docs()
#         }
#
#         import json
#         with open(os.path.join(output_dir, 'deploy_config.json'), 'w') as f:
#             json.dump(config, f, indent=2)
#
#         # 3. 创建启动脚本
#         self._create_startup_script(output_dir)
#
#         print(f"✅ 部署包已创建: {output_dir}")
#         return output_dir