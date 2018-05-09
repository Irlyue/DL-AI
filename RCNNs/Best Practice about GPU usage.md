# Best Practice about GPU usage

1. 使用环境变量限制只有某些GPU对框架可见

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python trian.py
   ```

2. 针对特定框架，在代码里限制GPU的使用（包括哪些GPU，GPU使用量等）

   ```Python
   #####################################(1)#############################################
   # in tensorflow
   with tf.device('/GPU:0'):
       # custom code
       
       
   ######################################(2)############################################  
   # running configuration
   ####################
   # Common Arguments #
   ####################
   # allow_soft_placement: default to True, if some operations are not provided in GPU mode, then it will automatically switch to CPU.
   # log_device_placement: just what it is

   # 1 use no GPUs
   session_config = {
       "allow_soft_placement": True, 
       "device_count": {
           "GPU": 0
       }
   }

   # 2 allocate only the amount of GPU menory needed, so TensorFlow won't take over all the GPU devices available.
   session_config = {
       "allow_soft_placement": True,
       "gpu_options": {
           "allow_growth": True
       }
   }

   config = tf.ConfigProto(**session_config)
   sess = tf.Session(config=config)
   # or
   config = tf.ConfigProto(**session_config)
   run_config = tf.estimator.RunConfig(session_config=config)
   estimator = tf.estimator.Estimator(config=run_config)
   ```

3. 一些常用的关于GPU的bash命令

   ```Bash
   # 1. list the memory useage of all GPUs
   nvidia-smi
   # 2. list memory usage of specific GPU
   nvidia-smi -i $GPU_ID
   # 3. list all the processes about GPUs
   # This is useful because sometimes many processes occupying the GPUs are not returned by nvidia-smi, you may need this command to decide which processes to kill instead of reboot.
   fuser -v /dev/nvidia*     # or sudo it to see all users
   ```

4. 一些杀进程的命令

   ```Bash
   kill $PID
   kill -u $USER_ID # kill all the processes owned by specific user
   kill %JOB_ID  # use command `jobs` to display background jobs
   ```

   ​