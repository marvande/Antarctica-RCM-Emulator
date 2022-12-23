[![LinkedIn][linkedin-shield]][linkedin-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/EPFLlogo.png" alt="Logo" width="150" height="80">
  </a>

  <h3 align="center">Deep Learning Regional Climate Model Emulators:</h3>

  <p align="center">
    a comparison of two downscaling training frameworks
    <br />
    <a href="https://github.com/marvande/master-thesis/blob/main/master_thesis.pdf"><strong>Explore the report »</strong></a>
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Regional climate models (RCMs) have a high computational cost due to their higher spatial resolution compared to global climate models (GCMs). Therefore, various downscaling approaches have been developed as a surrogate for the dynamical downscaling of GCMs. This study assesses the potential of using a cost-efficient machine learning alternative to dynamical downscaling by using the example case study of emulating surface mass balance (SMB) over the Antarctic Peninsula. More specifically, we determine the impact of the training framework by comparing two training scenarios: (1) a perfect and (2) an imperfect model framework. In the perfect model framework, the RCM-emulator learns only the downscaling function, similar to classical super-resolution approaches; therefore, it was trained with upscaled RCM features at GCM resolution. This emulator accurately reproduced SMB when evaluated on upscaled RCM features (mean RMSE of 0.27 mm w.e./day), but its predictions on GCM data conserved RCM-GCM inconsistencies and led to underestimation. In the imperfect model framework, the RCM-emulator was trained with GCM features and downscaled the GCM while exposed to RCM-GCM inconsistencies. This emulator predicted SMB close to the truth, showing it learned the underlying inconsistencies and dynamics. Our results suggest that a deep learning RCM-emulator can learn the proper GCM to RCM downscaling function while working directly with GCM data. Furthermore, the RCM-emulator presents a significant computational gain compared to an RCM simulation. We conclude that machine learning emulators can be applied to produce fast and fine-scaled predictions of RCM simulations from GCM data.



### Built With

* [![PyTorch][pytorch.py]][pytorch-url]
* [![Python][python.py]][python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

All notebooks to run the model and explore its pre-processing are available [here](https://github.com/marvande/master-thesis/tree/main/scr). 



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Marijn van der Meer - marijnvandermeer@bluewin.ch

Project Link: [https://github.com/marvande](https://github.com/marvande)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/marvande/master-thesis/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/marvande/master-thesis/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/marvande/master-thesis/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/marvande/master-thesis/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/marijn-van-der-meer/
[product-screenshot]: images/screenshot.png
[pytorch-url]: https://pytorch.org/
[pytorch.py]: https://img.shields.io/badge/PyTorch-0769AD?style=for-the-badge&logo=PyTorch&logoColor=white
[python-url]: https://www.python.org/
[python.py]: https://img.shields.io/badge/Python-563D7C?style=for-the-badge&logo=python&logoColor=white
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
