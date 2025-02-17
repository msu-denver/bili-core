::Set LF as your line ending default.
git config --global core.eol lf

::Set autocrlf to false to stop converting between windows style (CRLF) and Unix style (LF)
git config --global core.autocrlf false

::Change to the directory of this script
cd %~dp0

::Remove the index and force Git to rescan the working directory.
rm ..\..\.git\index

::Rewrite the Git index to pick up all the new line endings.
git reset
