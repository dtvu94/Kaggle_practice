import numpy as np


# integrated gradients
def integrated_gradients(input_img, 
                        model, 
                        label_pred, 
                        predict_and_gradient, 
                        baseline, 
                        steps):
    if baseline is None:
        baseline = 0 * input_img 
    # scale input_img and compute gradients
    list_grads = []
    for i in range(0, steps + 1):
        tmp_input = baseline + (float(i) / steps) * (input_img - baseline)
        tmp_input = np.array(np.float32(tmp_input))
        _, grad = predict_and_gradient(tmp_input, model, label_pred)
        list_grads.append(grad)
    list_grads = np.array(list_grads)
    average_gradients = np.average(list_grads[:-1], axis=0)
    integrated_grad = (input_img - baseline) * average_gradients
    # from shape (1, 3, x, y) -> (x, y, 3)
    integrated_grad = np.squeeze(integrated_grad, axis=0)
    integrated_grad = np.transpose(integrated_grad, (1, 2, 0))
    return integrated_grad

def random_baseline_integrated_gradients(input_img, 
                                        model, 
                                        label_pred, 
                                        predict_and_gradient, 
                                        steps, 
                                        number_trial):
    list_integ_gradients = []
    for i in range(number_trial):
        tmp_baseline = 255.0 *np.random.random(input_img.shape)
        integrated_grad = integrated_gradients(input_img, 
                                                model, 
                                                label_pred, 
                                                predict_and_gradient,
                                                tmp_baseline, 
                                                steps)
        
        list_integ_gradients.append(integrated_grad)
        print('Done trial {}'.format(i))
    average_integ_gradients = np.average(np.array(list_integ_gradients), axis=0)
    return average_integ_gradients
