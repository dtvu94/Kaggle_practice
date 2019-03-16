import numpy as np


# integrated gradients
def integrated_gradients(input_img, 
                        model, 
                        label_idx, 
                        predict_and_gradients, 
                        baseline, 
                        steps):
    if baseline is None:
        baseline = 0 * input_img 
    # scale input_img and compute gradients
    scaled_input_img = [baseline + (float(i) / steps) * (input_img - baseline) \
                        for i in range(0, steps + 1)]
    
    grads, _ = predict_and_gradients(scaled_input_img, model, label_idx)
    
    average_gradients = np.average(grads[:-1], axis=0)
    average_gradients = np.transpose(average_gradients, (1, 2, 0))
    
    integrated_grad = (input_img - baseline) * average_gradients
    return integrated_grad

def random_baseline_integrated_gradients(input_img, 
                                        model, 
                                        label_idx, 
                                        predict_and_gradients, 
                                        steps, 
                                        number_trial):
    list_integ_gradients = []
    for i in range(number_trial):
        tmp_baseline = 255.0 *np.random.random(input_img.shape)
        integrated_grad = integrated_gradients(input_img, 
                                                model, 
                                                label_idx, 
                                                predict_and_gradients,
                                                tmp_baseline, 
                                                steps)
        
        list_integ_gradients.append(integrated_grad)
        print('Done trial {}'.format(i))
    average_integ_gradients = np.average(np.array(list_integ_gradients), axis=0)
    return average_integ_gradients
