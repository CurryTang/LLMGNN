args = get_command_line_args()    
params_dict = load_yaml(args.yaml_path)
data_path = params_dict['DATA_PATH']
if args.mode == "main":
    main(data_path, args = args)