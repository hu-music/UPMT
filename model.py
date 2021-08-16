import tensorflow as tf
import numpy as np
import miditoolkit
import modules
import pickle
import utils
import time
from collections import Counter
import numpy as np
from glob import glob
import random
class PopMusicTransformer(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, checkpoint, is_training=False):
        # load dictionary
        self.dictionary_path = '{}/dictionary.pkl'.format(checkpoint)
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        # model settings
        self.group_size = 4 
        self.mem_len = 128
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 0.0002
        # load model
        self.is_training = is_training
        if self.is_training:
            self.batch_size = 2 
        else:
            self.batch_size = 1
        self.checkpoint_path = '{}/model'.format(checkpoint)
        self.load_model()

    ########################################
    # load model
    ########################################
    def load_model(self):
        # placeholders
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        if self.is_training:
            self.J = tf.compat.v1.placeholder(tf.float32, shape=[308])
            self.bword = tf.compat.v1.placeholder(tf.float32, shape=[None])
        else:
            self.bword=[0.01]*308
            self.J= [0.01]*308
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
        # model
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01, seed=None)
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            loss, self.logits, self.new_mem= modules.transformer(
                dec_inp=xx,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=self.dropout,
                dropatt=self.dropout,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self.is_training,
                J=self.J,
                bword=self.bword,
                mem_len=self.mem_len,
                cutoffs=[],
                div_val=-1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True)
            print('transformer loss',loss)

        self.avg_loss = tf.reduce_mean(loss)
        # vars
        all_vars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(self.avg_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))
        all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        # optimizer
        decay_lr = tf.compat.v1.train.cosine_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=400000,
            alpha=0.004)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr)
        self.train_op = optimizer.apply_gradients(grads_and_vars, self.global_step)
        # saver
        self.saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.saver.restore(self.sess, self.checkpoint_path)

    ########################################
    # temperature sampling
    ########################################
    def temperature_sampling(self, logits, temperature, topk):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction
    ########################################
    # extract events for prompt continuation
    ########################################
    def extract_events_all(self, input_path):
        note_items, tempo_items = utils.read_items_all(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

    def extract_events(self, input_path):  # events from tracks except the one with most note informaiton
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
    def extract_events_single(self, input_path):
        note_items, tempo_items = utils.read_items_single(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
    def pitchclass(self,noteon_temp):
        pitch_class=[]
        for i in range(len(noteon_temp)):
            if noteon_temp[i]<12:
                pitch_class.append(-1)
            elif 12<=noteon_temp[i]<24:
                pitch_class.append(0)
            elif 24<=noteon_temp[i]<36:
                pitch_class.append(1)
            elif 36<=noteon_temp[i]<48:
                pitch_class.append(2)
            elif 48<=noteon_temp[i]<60:
                pitch_class.append(3)
            elif 60<=noteon_temp[i]<72:
                pitch_class.append(4)
            elif 72<=noteon_temp[i]<84:
                pitch_class.append(5)
            elif 84<=noteon_temp[i]<96:
                pitch_class.append(6)
            elif 96<=noteon_temp[i]<108:
                pitch_class.append(7)
            elif 108<=noteon_temp[i]<120:
                pitch_class.append(8)
            elif 120<=noteon_temp[i]<128:
                pitch_class.append(9)
        return pitch_class
    def mid2seq_all(self,prompt):
        events = self.extract_events_all(prompt)
        words=[[]]
        noteon_temp=[]
        noted_temp=[]
        position_temp=[]
        #fix some events that are not in the original dictionary 
        for e in events:
            if e.name == 'Note Velocity' and e.value >21:
                e.value =21
            if e.name == 'Note Velocity' and e.value <3:
                e.value =3
            if e.name == 'Tempo Value'and e not in self.event2word:
                e.value = 59  
            if e.name == 'Note On' and e.value >= 108:
                e.value = 107
            if e.name=="Note On":
                noteon_temp.append(e.value)  
            if e.name=="Note Duration":
                noted_temp.append(self.event2word['{}_{}'.format(e.name, e.value)])  
            if e.name=="Position":
                position_temp.append(self.event2word['{}_{}'.format(e.name, e.value)])     
            words[0].append(self.event2word['{}_{}'.format(e.name, e.value)])
        words[0].append(self.event2word['Bar_None'])
        return words, noteon_temp,noted_temp,position_temp

    def mid2seq(self,prompt):
        events = self.extract_events(prompt)
        words=[[]]
        noteon_temp=[]
        noted_temp=[]
        position_temp=[]
        #fix some events that are not in the original dictionary 
        for e in events:
            if e.name == 'Note Velocity' and e.value >21:
                e.value =21
            if e.name == 'Note Velocity' and e.value <3:
                e.value =3
            if e.name == 'Tempo Value'and e not in self.event2word:
                e.value = 59  
            if e.name == 'Note On' and e.value >= 108:
                e.value = 107                
            if e.name=="Note On":
                noteon_temp.append(e.value)  
            if e.name=="Note Duration":
                noted_temp.append(self.event2word['{}_{}'.format(e.name, e.value)])  
            if e.name=="Position":
                position_temp.append(self.event2word['{}_{}'.format(e.name, e.value)])     
            words[0].append(self.event2word['{}_{}'.format(e.name, e.value)])
        words[0].append(self.event2word['Bar_None'])
        return words, noteon_temp,noted_temp,position_temp


    def mid2seq_single(self,prompt):
        events = self.extract_events_single(prompt)
        words=[[]]
        noteon_temp=[]
        noted_temp=[]
        position_temp=[]
        #fix some events that are not in the original dictionary 
        for e in events:
            if e.name == 'Note Velocity' and e.value >21:
                e.value =21
            if e.name == 'Note Velocity' and e.value <3:
                e.value =3
            if e.name == 'Tempo Value'and e not in self.event2word:
                e.value = 59  
            if e.name == 'Note On' and e.value >= 108:
                e.value = 107
            if e.name=="Note On":
                noteon_temp.append(e.value)  
            if e.name=="Note Duration":
                noted_temp.append(self.event2word['{}_{}'.format(e.name, e.value)])  
            if e.name=="Position":
                position_temp.append(self.event2word['{}_{}'.format(e.name, e.value)])     
            words[0].append(self.event2word['{}_{}'.format(e.name, e.value)])
        words[0].append(self.event2word['Bar_None'])
        return words, noteon_temp,noted_temp,position_temp
    # generate
    ########################################
    def generate_noteon(self, temperature, topk, output_path,smpi=np.array([-2,1,1,3]),prompt=None,modes='single'): # transfer position
        noteon_idx=[]
        notev_idx=[]
        for i in range(len(self.word2event)):
            if self.word2event[i].split('_')[0]=='Note On':
                noteon_idx.append(i)
            elif self.word2event[i].split('_')[0]=='Note Velocity':
                notev_idx.append(i)
        if modes == 'single':
            ww=[]
            words,noteon_temp,_,_= self.mid2seq_single(prompt) # transfer on this track: the track with most note information
            words1,noteon_temp1,_,_= self.mid2seq(prompt) # these tracks remains same, with a lower volumn
            for i in range(len(words1[0])):
                if words1[0][i] in notev_idx:
                    words1[0][i] = 57
            bars=np.where(np.array(words[0])==0)[0]
            bars1=np.where(np.array(words1[0])==0)[0]
        else:
            words,noteon_temp,_,_= self.mid2seq_all(prompt)

        tem=[] # all note on in input sequence 
        for i in range(len(words[0])):
            if words[0][i] in noteon_idx:
                tem.append(i)             

    # input
        traindata=[[]]
        traindata[0]=words[0][:tem[0]]
        original_length = len(traindata[0])
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        # generate
        initial_flag = 1
        temi=tem[0]
        noteoncount=0
        words_temp=[]
        idx=0
        while temi <len(words[0]):
            if temi in tem:  
                # input
                if initial_flag:
                    temp_x = np.zeros((self.batch_size, original_length))
                    for b in range(self.batch_size):
                        for z, t in enumerate(traindata[b]):
                            temp_x[b][z] = t
                    initial_flag = 0
                else:
                    temp_x = np.zeros((self.batch_size, 1))
                    for b in range(self.batch_size):
                        temp_x[b][0] = traindata[b][-1]
                # prepare feed dict
                feed_dict = {self.x: temp_x}
                for m, m_np in zip(self.mems_i, batch_m):
                    feed_dict[m] = m_np
                _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
                # sampling
                _logit = _logits[-1, 0]
                if words_temp==[]: # predict by Transformer-XL
                    _logit= _logit[noteon_idx]  
                    word = self.temperature_sampling(
                            logits=_logit, 
                            temperature=temperature,
                            topk=topk)
                    word = noteon_idx[word]
                    words[0][temi] = word
                    print('word:',words[0][temi])
                    traindata[0].append(word)
                else: #predict by SMPI
                    words[0][temi] = words_temp[idx]
                    print('word:',words[0][temi])
                    traindata[0].append(words_temp[idx])
                    idx+=1
                    if idx >len(words_temp)-1:
                        idx=0
                        words_temp=[]
                #Update new noteone sequence based on SMPI
                if words_temp==[] and int(self.word2event[words[0][tem[noteoncount]]].split('_')[1]) - int(self.word2event[words[0][tem[noteoncount-1]]].split('_')[1]) == smpi[0]: #if match the first item in SMPI
                    for id1 in range(1,len(smpi)):
                        try:
                            word = self.event2word['Note On_'+str(int(self.word2event[word].split('_')[1]) +smpi[id1])]
                            words_temp.append(word)
                        except:
                            print('note on out of range!')
                            continue
                noteoncount +=1 
            else:
                if initial_flag:
                    temp_x = np.zeros((self.batch_size, original_length))
                    for b in range(self.batch_size):
                        for z, t in enumerate(traindata[b]):
                            temp_x[b][z] = t
                    initial_flag = 0
                else:
                    temp_x = np.zeros((self.batch_size, 1))
                    for b in range(self.batch_size):
                        temp_x[b][0] = traindata[b][-1]
                # prepare feed dict
                feed_dict = {self.x: temp_x}
                for m, m_np in zip(self.mems_i, batch_m):
                    feed_dict[m] = m_np 
                _, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
                words[0][temi] = words[0][temi]
                traindata[0].append(words[0][temi])
            temi+=1
            # re-new mem
            batch_m = _new_mem
        if modes=='single':
            i=0
            ww=[]
            while i < min(len(bars),len(bars1))-1:
                ww=ww+words[0][bars[i]:bars[i+1]]+words1[0][bars1[i]+1:bars1[i+1]]
                i+=1
        else:
            ww=words[0]
        # write
        utils.write_midi(
            words=ww,
            word2event=self.word2event,
            output_path=output_path,
            prompt_path=None)
   
    ########################################
    # prepare training data
    ########################################

    def prepare_data(self, midi_paths):
        length=self.mem_len
        noteon_position=[]
        for i in range(len(self.word2event)):
            if self.word2event[i].split('_')[0] in ['Note On']:
                noteon_position.append(i)
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events_all(path)
            all_events.append(events)
        # event to word
        all_words = []
        iii=0
        for events in all_events:
            words = []
            for event in events:
                if event.name in ['Bar','Position','Note On']:
                    e = '{}_{}'.format(event.name, event.value)
                    if e in self.event2word:
                        words.append(self.event2word[e])
                    else:
                        # OOV
                        if event.name == 'Note Velocity':
                            # replace with max velocity based on our training data
                            words.append(self.event2word['Note Velocity_21'])
                        elif event.name == 'Tempo Value':
                            if event.value > 54:
                                words.append(self.event2word['Tempo Value_59'])
                            elif event.value <20:
                                words.append(self.event2word['Note Velocity_16'])
                            else:
                                words.append(self.event2word['Note Velocity_36'])
                        else:
                            # something is wrong
                            # you should handle it for your own purpose
                            print('something is wrong! {}'.format(e),iii)
            index=1+np.where(np.diff(words)==0)[0]
            words=np.delete(words,index)
            all_words.append(words)
            iii+=1
        # to training data
        self.group_size = 5 
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0,len(words)-length-1,length):
                x=words[i:i+length]
                y=words[i+1:i+length+1]
                pairs.append([x, y])
            pairs = np.array(pairs)
            print('pairs #:',pairs.shape)
            # abandon the last
            for i in np.arange(0, len(pairs)-self.group_size,self.group_size*2):
                data = pairs[i:i+self.group_size]
                if len(data) == self.group_size:
                    segments.append(data)
        segments = np.array(segments)
        print('segments #:',segments.shape)
        pickle.dump(segments, open('./test/data/training.pickle',"wb"), protocol=2)
        return segments    
    ########################################
    # finetune
    ########################################
    def finetune(self, training_data,favoritepath,alpha,output_checkpoint_folder):
        # shuffle
        Jnew1= [alpha]*308
        words,noteon_temp,noted_temp,position_temp= self.mid2seq(favoritepath)
        bwordnew=list(Counter(noteon_temp).keys())
        bwordnew=[self.event2word['Note On_'+ str(bwordnew[i])] for i in range(len(bwordnew))]

        Jnew=[list(Counter(noteon_temp).values())[i]/len(noteon_temp) for i in range(len(list(Counter(noteon_temp).values())))]
        Jnew=[Jnew[i]+alpha for i in range(len(Jnew))]
        for i in range(len(bwordnew)):
            Jnew1[bwordnew[i]] = Jnew[i]      
        Jnew=Jnew1
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        num_batches = len(training_data) // self.batch_size
        print('num_batches:',num_batches)
        print('Jnew:',Jnew)
        print('Jnew ratio:',[Jnew[i]/alpha for i in range(len(Jnew))])
        loss_ct=[]
        st = time.time()
        for e in range(201):
            total_loss = []
            for i in range(num_batches):
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    feed_dict = {self.x: batch_x, self.y: batch_y,self.J:Jnew,self.bword:bwordnew}
                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np  #m_np = batch_m: mem_len*bs*d_model
                    # run
                    logits,_, gs_, loss_, new_mem_ = self.sess.run([self.logits,self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    batch_m = new_mem_

                    total_loss.append(loss_)
                    loss_ct.append(loss_)
                    # print('logits:',logits)
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))
            self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
            # stop
            if np.mean(total_loss) <= 0.1:
                # self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
                break
        return loss_ct

    ########################################
    # close
    ########################################
    def close(self):
        self.sess.close()



    


