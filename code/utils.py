def plot_topwords(args):

    wv= np.load(args.embded_file)
    vocabulary = np.load(args.vocab).item()
    
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printargs(suppress=True)
    Y = tsne.fit_transform(wv[1:1001,:])
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')    

def weighting_scheme(batch_size, batch_n, epochs, data_size, weighting_scheme):
    if weighting_scheme == 1:
        return batch_size / data_size
    elif weighting_scheme == 2:
        M = epochs * data_size
        return np.exp(-(1-.1**(100))*(batch_n-1))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, best_model_file):

    if os.path.isfile(best_model_file):

        print("=> loading checkpoint '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(best_model_file, checkpoint['epoch']))
        print("best validation accuracy is %s" % best_val_acc)
    else:
        print("=> no checkpoint found at '{}'".format(best_model_file))

    return model, optimizer, best_loss, epoch
