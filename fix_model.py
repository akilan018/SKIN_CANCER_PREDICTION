import h5py

def fix_keras3_model_for_keras2(model_path):
    print(f"Opening {model_path} to manually remove Keras 3 metadata...")
    try:
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                import json
                
                # Load the config
                config_str = f.attrs.get('model_config')
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config = json.loads(config_str)
                
                changed = False
                # The issue is specifically the 'batch_shape' argument
                if 'config' in config and 'layers' in config['config']:
                    for layer in config['config']['layers']:
                        if layer['class_name'] == 'InputLayer':
                            if 'batch_shape' in layer['config']:
                                print(f"Found Keras 3 'batch_shape': {layer['config']['batch_shape']}")
                                layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                                changed = True
                
                if changed:
                    print("Fixed the InputLayer config. Saving back to HDF5 metadata...")
                    new_config_str = json.dumps(config)
                    f.attrs.modify('model_config', new_config_str.encode('utf-8'))
                    print("Done! This model is now compatible with TensorFlow 2.15 (Keras 2).")
                else:
                    print("No Keras 3 'batch_shape' found. Model might already be Keras 2.")
            else:
                print("No model_config found in attributes.")
    except Exception as e:
        print(f"Error modifying HDF5 attributes: {e}")

if __name__ == '__main__':
    fix_keras3_model_for_keras2('best_model.h5')
