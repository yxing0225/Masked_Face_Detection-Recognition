


if __name__ == "__main__":
    train_img_dir = r"F:\dataset\CASIA\CASIA_test"
    # train_img_dir = [r"F:\dataset\CASIA\CASIA-WebFace_aligned",
    #                  r"F:\dataset\CASIA\CASIA-WebFace_aligned(mask)"]
    test_img_dir = r"F:\dataset\FLW_detect_aligned"
    # test_img_dir = r"D:\dataset\lfw_2\detect_aligned"
    label_dict = None
    embed_length = 128

    para_dict = {"train_img_dir":train_img_dir,"test_img_dir":test_img_dir,"label_dict":label_dict}

    cls = Facenet(para_dict)

    model_shape = [None,80,80,3]
    infer_method = "inception_resnet_v1"
    loss_method = "cross_entropy"
    opti_method = "adam"
    learning_rate = 5e-4
    save_dir = r"F:\model_saver\test_xx"

    para_dict = {"model_shape":model_shape,"infer_method":infer_method,"loss_method":loss_method,
                 "opti_method":opti_method,'learning_rate':learning_rate,"save_dir":save_dir,'embed_length':embed_length}
    cls.model_init(para_dict)

    epochs = 100
    GPU_ratio = None
    batch_size = 12
    ratio = 1.0
    para_dict = {'epochs':epochs, "GPU_ratio":GPU_ratio, "batch_size":batch_size,"ratio":ratio}

    cls.train(para_dict)