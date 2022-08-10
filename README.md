[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/EPFLlogo.png" alt="Logo" width="150" height="80">
  </a>

  <h3 align="center">Regional Surface Mass Balance Emulator based on Deep Learning</h3>

  <p align="center">
    Local-scale SMB predictions over the Antarctic Peninsula
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
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The Antarctic ice sheet contains the majority of ice on Earth and would be the most significant
contributor to sea-level rise if it were to melt completely. A critical variable in studying the
stability and evolution of ice on Antarctica, such as glaciers and ice shelves, is surface mass
balance (SMB). SMB is the balance between processes that add and remove ice on top of an ice
sheet, glacier, or ice shelf. For example, those processes can include snowfall, surface melt, or
erosion by winds. To predict changes in the Antarctic ice sheet, climate scientists need highly
detailed information about SMB.

High-resolution maps of SMB are usually only produced by regional climate models (RCM).
But these are costly and time-consuming to create because they require calculations on very
powerful computers. The goal of this project was to accomplish this faster and with similar
accuracy by using machine learning to predict SMB changes.

We trained a machine-learning algorithm i.e., a computer program that recognizes patterns in
data, to learn the relationship between a group of low-resolution images of climate variables
and a high-resolution image of regional SMB. The low-resolution data consisted of a bundle
of atmospheric variables influencing SMB, such as precipitation, temperature, and surface
pressure over the Antarctic Peninsula, an interesting region of Antarctica. Those coarse-grain
atmospheric variables are easier and cheaper to obtain from worldwide simulations made by
climate scientists.

Our machine-learning model can recreate regional images of SMB almost identical to ground
truth observations by learning the relationship between low and high-resolution inputs. The
model only struggles to predict proper SMB values for very dry areas (where the SMB is small)
because it tends to emphasize points with larger changes in SMB. The predictive model is very
fast, with training under ten minutes and output in a few seconds. What remains to be studied is
whether the model would need new training for every new (unseen) region in Antarctica or if it
could be deployed across areas.

We conclude that it is possible to make fast and detailed reproductions of SMB at regional scales
using machine learning. Therefore, machine learning can be an interesting and cheap tool for
gathering local-scale information about how ice sheets vary with climate change.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![PyTorch][pytorch.py]][pytorch-url]
* [![Python][python.py]][python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Marijn van der Meer - marijnvandermeer@bluewin.ch

Project Link: [https://github.com/marvande](https://github.com/marvande)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

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
