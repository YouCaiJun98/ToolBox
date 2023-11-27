* Take the following steps to set up my new environment:
* tmux environment:
  * copy the tmux config file to `~/`:

  '''bash
  cp .tmux.conf ~/
  '''

* add alias to ~/.bashrc:
  * cp lines in `.bashrc` & do `source ~/.bashrc`
* set up vim:
  * install independences

   '''bash
    sudo apt-get install silversearcher-ag ctags
    sudo apt-get install gcc g++ cmake
    sudo apt-get install libncurses5-dev # 插件需要的软件包
    wget http://mirrors.ustc.edu.cn/gnu/global/global-6.4.tar.gz
    tar xf global-6.4.tar.gz
    cd global-6.4
    ./configure && make && sudo make install
   '''
  * cp `.vimrc` to `~/`
  * open `~/.vimrc` and conduct `: PluginInstall` to install the plugins
  * fix some bug:

    '''bash
    ~/.vim/bundle/molokai/colors/molokai.vim:line 132:none -> NONE
    '''
