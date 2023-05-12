# Pytorch-GAN
使用Pytorch实现GAN 的过程
<hr></hr>

<div>
	<p><a href="https://mydreamambitious.blog.csdn.net/article/details/130362296"> GAN原理及Pytorch框架实现GAN（比较容易理解）</a></p>
	<p><a href="https://mydreamambitious.blog.csdn.net/article/details/130402245"> Pytorch框架实现DCGAN（比较容易理解）</a></p>
	<p><a href="https://mydreamambitious.blog.csdn.net/article/details/130447595"> CycleGAN的基本原理以及Pytorch框架实现</a></p>
	<p><a href="https://mydreamambitious.blog.csdn.net/article/details/130471766"> WGAN基本原理及Pytorch实现WGAN</a></p>
	<p><a href="https://mydreamambitious.blog.csdn.net/article/details/130483339"> Pytorch框架实现WGAN-GP</a></p>
	<p><a href="https://mp.csdn.net/mp_blog/creation/editor/130506045">Pytorch框架实现Pix2Pix(Image-to-image)</a></p>
	<p>自编码器开始了(AutoEncoder)......</p>
	<p><a href="https://mydreamambitious.blog.csdn.net/article/details/130623570">Pytorch实现自编码器变种</a></p>
	<p><a href="https://mydreamambitious.blog.csdn.net/article/details/130541755">自编码器原理及使用Pytorch框架实现（AutoEncoder）</a></p>
</div>

<label>还在不断更新中……</label>
<hr></hr>
<div>
  <label><strong>Git上传文件代码到GitHub</strong></label>
  <ul>
    <li>第一步：新建一个空文件夹，用来上传文件: 比如 demo(文件夹中可以包含要上传的项目)</li>
    <li>第二步：进入新建的文件夹demo目录下，点进去空文件夹，鼠标右键，使用Git Bash Here 打开</li>
    <li>第三步：输入 git init ，初始化，在本地创建一个Git仓库</li>
    <li>第四步：输入 git add . 将项目添加到暂存区:
         <ul> 
            <li>注意： . 前面有空格，代表添加所有文件</li>
            <li>若添加单个文件输入：git add xxxx.xx（xxxx.xx为文件名）</li>
         </ul>
    </li>
    <li>第五步：输入 git commit -m "注释内容" 将项目提交到Git仓库</li>
    <li>第六步：输入 git branch -M main ，上传到 main 分支(这里必须确保Github上的创库已经建好)</li>
    <li>第七步：输入：git remote add origin https://github.com/xxxxx/test.git，和远程仓库连接</li>
    <li>第八步：输入 git push -u origin main 将本地项目推送到远程仓库(这一步可能提示：! [rejected]        main -> main (fetch first)
                error: failed to push some refs to 'https://github.com/KeepTryingTo/Pytorch-GAN.git')
                <label>
                     <strong>报错的原因是因为，每个仓库都有一个分支，也可以理解为大仓库里的小仓库，我们只是跟线上远程仓库有了关联，但没有跟线上远程仓库的某个分支关联，所以我                             们没法提交</strong>
                </label>
                <label>解决方案如下：</label>
                <ul>
                     <li>终端输入 git pull --rebase origin main 即可跟刚创建的线上远程仓库的默认分支main关联</li>
                     <li>再执行一下 git push -u origin main 即可将我们的项目文件上传到关联的线上远程文件中</li>
                </ul>
    </li>
  </ul>
</div>
