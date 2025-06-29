import librosa
import numpy as np
import matplotlib.pyplot as plt
import datetime
import fire
import os
import music21 as m21


def test():
    # welcome
    print('欢迎使用音频节奏与节拍分析工具')
    print('输入文件路径后按回车键开始分析')

    tmp_path = input(':')
    # if not input
    if not tmp_path:
        print('未输入文件路径，程序退出')
        return


    # if file not exists
    if not os.path.exists(tmp_path):
        print('文件路径不存在，请检查后重新输入')
        return
    

    timeinfo = datetime.datetime.now()
    timeinfo = timeinfo.strftime('%Y-%m-%d %H:%M:%S')
    save_path = f'./img/节奏与节拍分析{timeinfo}.png'
    print(f'文件路径是:{tmp_path}')
    print(f'分析结果将保存到:{save_path}')
    print('正在分析音频节奏与节拍信息...')
    
    # 音频载入分析
    y, sr = librosa.load(tmp_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # 节奏与节拍信息显示
    hop_length = 512
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                             y_axis='mel', x_axis='time', hop_length=hop_length,
                             ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title=f'{tmp_path}\n')
    ax[1].plot(times, librosa.util.normalize(onset_env),
             label='Rhythm peaks')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
               linestyle='--', label='Beats')
    ax[1].legend()



    # 乐谱化信息
    s = m21.stream.Stream()
    # 设置乐谱的速度
    s.insert(0, m21.tempo.MetronomeMark(number=tempo))
    # 设置乐谱的拍号
    s.insert(0, m21.meter.TimeSignature('4/4'))
    for i in onset_env:
        if i > 0:
            note = m21.note.Note(quarterLength=0.125)
        else:
            note = m21.note.Rest(quarterLength=0.125)
        s.append(note)
    # 保存乐谱为MIDI文件
    midi_path = f'./midi/节奏与节拍分析{timeinfo}.mid'
    s.write('midi', fp=midi_path)
    print(f'乐谱已保存到: {midi_path}')
    #
    s.show()



    # save img to local
    plt.savefig(save_path)
    print('分析结果已保存到: {save_path}')

    # pop up window
    print('显示分析结果...')
    plt.show()
    print('分析完成！')
    


if __name__ == '__main__':
    if not os.path.exists('./img'):
        os.makedirs('./img')
    if not os.path.exists('./midi'):
        os.makedirs('./midi')
    fire.Fire(test)



