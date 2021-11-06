import os,math,cv2,shutil
import numpy as np
import tensorflow

#print("Tensorflow version: ",tf.__version__)

img_format = {'png','jpg','bmp'}

def model_restore_from_pb(pb_path, node_dict,GPU_ratio=None):
    tf_dict = dict()
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,
                                )
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())
        for key, value in node_dict.items():
            try:
                node = sess.graph.get_tensor_by_name(value)
                tf_dict[key] = node
            except:
                print("node:{} does not exist in the graph".format(key))
        return sess, tf_dict

def img_removal_by_embed(root_dir,output_dir,pb_path,node_dict,threshold=0.7,type='copy',GPU_ratio=None, dataset_range=None):
    img_format = {"png", 'jpg', 'bmp'}
    batch_size = 64

    # collect all folders
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No sub-dirs in ", root_dir)
    else:
        #dataset range
        if dataset_range is not None:
            dirs = dirs[dataset_range[0]:dataset_range[1]]

        # model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
        tf_input = tf_dict['input']
        tf_phase_train = tf_dict['phase_train']
        tf_embeddings = tf_dict['embeddings']
        model_shape = [None, 160, 160, 3]
        feed_dict = {tf_phase_train:False}
        # feed_dict[tf_phase_train] = False

        # #get the model shape
        # if tf_input.shape[1].value is None:
        #     model_shape = (None, 160, 160, 3)
        # else:
        #     model_shape = (None, tf_input.shape[1].value, tf_input.shape[2].value, 3)
        # print("The mode shape of face recognition:", model_shape)
        #
        # # set the feed_dict
        # feed_dict = dict()
        # if 'keep_prob' in tf_dict.keys():
        #     tf_keep_prob = tf_dict['keep_prob']
        #     feed_dict[tf_keep_prob] = 1.0
        # if 'phase_train' in tf_dict.keys():
        #     tf_phase_train = tf_dict['phase_train']
        #     feed_dict[tf_phase_train] = False

        # tf setting for calculating distance
        with tf.Graph().as_default():
            tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
            tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
            tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
            # GPU setting
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,  # 可以转化
                                    )
            config.gpu_options.allow_growth = True
            sess_cal = tf.Session(config=config)
            sess_cal.run(tf.global_variables_initializer())

        #process each folder
        for dir_path in dirs:
            paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            len_path = len(paths)
            if len_path == 0:
                print("No images in ",dir_path)
            else:
                # create the sub folder in the output folder
                save_dir = os.path.join(output_dir, dir_path.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # calculate embeddings
                ites = math.ceil(len_path / batch_size)
                embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
                for idx in range(ites):
                    num_start = idx * batch_size
                    num_end = np.minimum(num_start + batch_size, len_path)
                    # read batch data
                    batch_dim = [num_end - num_start]#[64]
                    batch_dim.extend(model_shape[1:])#[64,160, 160, 3]
                    batch_data = np.zeros(batch_dim, dtype=np.float32)
                    for idx_path,path in enumerate(paths[num_start:num_end]):
                        img = cv2.imread(path)
                        if img is None:
                            print("Read failed:",path)
                        else:
                            img = cv2.resize(img, (model_shape[2], model_shape[1]))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            batch_data[idx_path] = img
                    batch_data /= 255  # norm
                    feed_dict[tf_input] = batch_data
                    embeddings[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)

                # calculate ave distance of each image
                feed_dict_2 = {tf_ref: embeddings}
                ave_dis = np.zeros(embeddings.shape[0], dtype=np.float32)
                for idx, embedding in enumerate(embeddings):
                    feed_dict_2[tf_tar] = embedding
                    distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                    ave_dis[idx] = np.sum(distance) / (embeddings.shape[0] - 1)
                #remove or copy images
                for idx,path in enumerate(paths):
                    if ave_dis[idx] > threshold:
                        print("path:{}, ave_distance:{}".format(path,ave_dis[idx]))
                        if type == "copy":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.copy(path,save_path)
                        elif type == "move":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.move(path,save_path)

def check_path_length(root_dir,output_dir,threshold=5):
    # var
    img_format = {"png", 'jpg'}

    # collect all dirs
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]

    if len(dirs) == 0:
        print("No dirs in ",root_dir)
    else:
        for dir_path in dirs:
            leng = len([file.name for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format])
            if leng <= threshold:
                corresponding_dir = os.path.join(output_dir,dir_path.split("\\")[-1])
                leng_corre = len([file.name for file in os.scandir(corresponding_dir) if file.name.split(".")[-1] in img_format])
                print("dir name:{}, quantity of origin:{}, quantity of removal:{}".format(dir_path.split("\\")[-1],leng,leng_corre))

def delete_dir_with_no_img(root_dir):
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No dirs in ",root_dir)
    else:
        for dir_path in dirs:
            leng = len([file.name for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format])
            if leng == 0:
                shutil.rmtree(dir_path)
                print("Deleted:",dir_path)



if __name__ == "__main__":
    #the pb model down directly online
#     root_dir = r"E:\PycharmProjects\faceNet_and_Data\CASIA-WebFace\CASIA-WebFace_aligned"
#     output_dir = r"E:\PycharmProjects\faceNet_and_Data\CASIA-WebFace\mislabeled"
#     pb_path = r"E:\PycharmProjects\data_cleaning\[proj]CASIA+data+cleaning\Model_20180402-114759\20180402-114759.pb"
#     node_dict = {'input': 'input:0',
#                  'phase_train': 'phase_train:0',
#                  'embeddings': 'embeddings:0',
#                  # 'keep_prob': 'keep_prob:0',
#                  }
#     dataset_range = None
#     img_removal_by_embed(root_dir, output_dir, pb_path, node_dict, threshold=1.25, type='move', GPU_ratio=0.25,
#                          dataset_range=dataset_range)

    # # ----check_path_length
    # root_dir = r"E:\PycharmProjects\faceNet_and_Data\CASIA-WebFace\CASIA-WebFace_aligned"
    # output_dir = r"E:\PycharmProjects\faceNet_and_Data\CASIA-WebFace\mislabeled"
    # check_path_length(root_dir, output_dir, threshold=3)

    #----delete_dir_with_no_img
    root_dir = r"E:\PycharmProjects\faceNet_and_Data\CASIA-WebFace\CASIA-WebFace_aligned"
    delete_dir_with_no_img(root_dir)


