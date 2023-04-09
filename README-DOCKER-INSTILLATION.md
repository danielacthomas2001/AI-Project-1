# Installing Docker on Windows with WSL 2
Follow these steps to install Docker Desktop with WSL 2 support on Windows:

-Download Docker Desktop for Windows from the Docker website.

-Install Docker Desktop by following the usual installation instructions. If your system supports WSL 2, Docker Desktop prompts you to enable it during installation. Enable WSL 2 to continue.

-Start Docker Desktop from the Windows Start menu.

-From the Docker menu, select Settings and then General.

-Select the Use WSL 2 based engine check box. This option should be enabled by default if you have installed Docker Desktop on a system that supports WSL 2.

-Select Apply & Restart.


## Enabling Docker support in WSL 2 distros
You can further enhance your developer experience by installing at least one Linux distribution in WSL 2 and enabling Docker support for it. Here's how:

-Ensure the distribution runs in WSL 2 mode. To check the WSL mode, run the command:
-Copy code
wsl.exe -l -v
-To upgrade an existing Linux distribution to WSL 2, run:
scss
-Copy code
wsl.exe --set-version (distro name) 2
-To set WSL 2 as the default version for future installations, run:
arduino
-Copy code
wsl.exe --set-default-version 2
-When Docker Desktop starts, go to Settings > Resources > WSL Integration.

-The Docker-WSL integration is enabled on your default WSL distribution. To change your default WSL distro, run the command:

arduino
Copy code
wsl --set-default <distro name>
For example, to set Ubuntu as your default WSL distro, run:

arduino
Copy code
wsl --set-default ubuntu
-Optionally, select any additional distributions you would like to enable the Docker-WSL integration on.

-Select Apply & Restart.

That's it! You can now use Docker with WSL 2 on Windows.
