




# Dont forget to add the secrets 

# You can automate cleanup before running new containers by adding this to your CI script:
docker system prune -af --volumes

# Grow the partition (this makes the OS aware of the added space):That’s /dev/xvda and partition 1 — make sure it matches your earlier df -h output (/dev/xvda1).
sudo growpart /dev/xvda 1

# Final Command to Expand the File System:

sudo xfs_growfs -d /

# ✅Then Recheck:
df -h

✅ What You Just Completed
Step	Description
🔧 Modified volume	Increased EBS volume size in AWS Console
💻 SSH commands	Ran growpart and xfs_growfs to resize the partition and file system
📦 Reclaimed space	Your EC2 instance now has room for Docker builds and ML deployment
🔁 Ready to rerun	You can now confidently trigger your GitHub Actions CI/CD pipeline again
Lets go